#!/usr/bin/env python3
import argparse
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path


INTERFACES = [
    "org.contikios.cooja.interfaces.Position",
    "org.contikios.cooja.interfaces.Battery",
    "org.contikios.cooja.contikimote.interfaces.ContikiVib",
    "org.contikios.cooja.contikimote.interfaces.ContikiMoteID",
    "org.contikios.cooja.contikimote.interfaces.ContikiRS232",
    "org.contikios.cooja.contikimote.interfaces.ContikiBeeper",
    "org.contikios.cooja.interfaces.RimeAddress",
    "org.contikios.cooja.interfaces.IPAddress",
    "org.contikios.cooja.contikimote.interfaces.ContikiRadio",
    "org.contikios.cooja.contikimote.interfaces.ContikiButton",
    "org.contikios.cooja.contikimote.interfaces.ContikiPIR",
    "org.contikios.cooja.contikimote.interfaces.ContikiClock",
    "org.contikios.cooja.contikimote.interfaces.ContikiLED",
    "org.contikios.cooja.contikimote.interfaces.ContikiCFS",
    "org.contikios.cooja.interfaces.Mote2MoteRelations",
    "org.contikios.cooja.interfaces.MoteAttributes",
]


def add_text(parent, tag, text):
    child = ET.SubElement(parent, tag)
    child.text = text
    return child


def add_motetype(sim, identifier, description, source, binary, defines):
    mt = add_text(sim, "motetype", "\n      org.contikios.cooja.contikimote.ContikiMoteType\n      ")
    add_text(mt, "identifier", identifier)
    add_text(mt, "description", description)
    add_text(mt, "source", f"[CONTIKI_DIR]/examples/rpl-dra-ml/{source}")
    cmd = "$(MAKE) TARGET=cooja clean\n"
    cmd += f"$(MAKE) -j$(CPUS) {binary}.cooja TARGET=cooja"
    if defines:
        cmd += f" DEFINES={defines}"
    add_text(mt, "commands", cmd)
    for interface in INTERFACES:
        add_text(mt, "moteinterface", interface)
    return mt


def add_mote(sim, mote_id, motetype, x, y):
    mote = ET.SubElement(sim, "mote")
    pos = ET.SubElement(mote, "interface_config")
    pos.text = "\n        org.contikios.cooja.interfaces.Position\n        "
    ET.SubElement(pos, "pos", {"x": f"{x:.3f}", "y": f"{y:.3f}"})
    mid = ET.SubElement(mote, "interface_config")
    mid.text = "\n        org.contikios.cooja.contikimote.interfaces.ContikiMoteID\n        "
    add_text(mid, "id", str(mote_id))
    add_text(mote, "motetype_identifier", motetype)


def positions(node_count, seed, width, height):
    rng = random.Random(seed)
    cols = math.ceil(math.sqrt(node_count))
    rows = math.ceil(node_count / cols)
    dx = width / max(cols - 1, 1)
    dy = height / max(rows - 1, 1)
    result = [(width / 2, height / 2)]
    for i in range(1, node_count):
        row = (i - 1) // cols
        col = (i - 1) % cols
        x = col * dx + rng.uniform(-8, 8)
        y = row * dy + rng.uniform(-8, 8)
        result.append((max(0, min(width, x)), max(0, min(height, y))))
    return result


def build(args):
    attackers = int(round((args.node_count - 1) * args.attack_ratio / 100.0))
    rng = random.Random(args.seed + args.attack_ratio * 1000)
    attacker_ids = set(rng.sample(range(2, args.node_count + 1), attackers))
    coords = positions(args.node_count, args.seed, args.width, args.height)

    root = ET.Element("simconf", {"version": "2022112801"})
    sim = ET.SubElement(root, "simulation")
    add_text(sim, "title", f"RPL DRA ML {args.attack_ratio}%")
    add_text(sim, "randomseed", str(args.seed))
    add_text(sim, "motedelay_us", "1000000")

    radio = ET.SubElement(sim, "radiomedium")
    radio.text = "\n      org.contikios.cooja.radiomediums.UDGM\n      "
    add_text(radio, "transmitting_range", str(args.tx_range))
    add_text(radio, "interference_range", str(args.interference_range))
    add_text(radio, "success_ratio_tx", str(args.tx_success))
    add_text(radio, "success_ratio_rx", str(args.rx_success))

    events = ET.SubElement(sim, "events")
    add_text(events, "logoutput", "40000")

    add_motetype(sim, "root", "RPL root", "root-node.c", "root-node", "")
    add_motetype(sim, "benign", "Benign RPL node", "benign-node.c", "benign-node", "")
    add_motetype(
        sim,
        "attacker",
        "Decreased-rank attacker",
        "attacker-node.c",
        "attacker-node",
        f"DRA_ATTACK=1,DRA_RANK_DECREMENT={args.rank_decrement}",
    )

    for mote_id in range(1, args.node_count + 1):
        motetype = "root" if mote_id == 1 else "attacker" if mote_id in attacker_ids else "benign"
        add_mote(sim, mote_id, motetype, coords[mote_id - 1][0], coords[mote_id - 1][1])

    script_plugin = ET.SubElement(root, "plugin")
    script_plugin.text = "\n    org.contikios.cooja.plugins.ScriptRunner\n    "
    cfg = ET.SubElement(script_plugin, "plugin_config")
    script = f"""TIMEOUT({args.duration_ms}, log.testOK());
log.log("META nodes={args.node_count} attackers={attackers} attack_ratio={args.attack_ratio} seed={args.seed} attacker_ids={','.join(map(str, sorted(attacker_ids)))}\\n");
while(true) {{
  YIELD();
  if(msg.startsWith("TRACE") || msg.startsWith("TX") || msg.startsWith("RX") || msg.startsWith("DRA_ADVERTISE")) {{
    log.log(time + " mote=" + id + " " + msg + "\\n");
  }}
}}"""
    add_text(cfg, "script", script)
    add_text(cfg, "active", "true")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(args.output, encoding="UTF-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-count", type=int, default=50)
    parser.add_argument("--attack-ratio", type=int, choices=[0, 10, 20, 30], default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=600000)
    parser.add_argument("--tx-range", type=float, default=100.0)
    parser.add_argument("--interference-range", type=float, default=110.0)
    parser.add_argument("--tx-success", type=float, default=1.0)
    parser.add_argument("--rx-success", type=float, default=1.0)
    parser.add_argument("--rank-decrement", type=int, default=512)
    parser.add_argument("--width", type=float, default=250.0)
    parser.add_argument("--height", type=float, default=250.0)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
