# Cooja RPL-DRA Experiment

This folder contains the Contiki-NG application and patch used to generate
the Cooja traces for the RPL decreased-rank attack benchmark.

## Setup

Clone Contiki-NG and initialize Cooja:

```bash
git clone https://github.com/contiki-ng/contiki-ng.git
cd contiki-ng
git submodule update --init tools/cooja
```

Use Java 21 and GNU Make 4 or newer. On macOS with Homebrew:

```bash
brew install openjdk@21 make
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"
```

Copy the experiment into Contiki-NG and apply the RPL-lite DIO patch:

```bash
cp -R /path/to/multilayer-rpl-dra/cooja/rpl-dra-ml examples/
git apply /path/to/multilayer-rpl-dra/cooja/patches/rpl-lite-dra-advertise.patch
```

## Generate Scenarios

```bash
for ratio in 10 20 30; do
  python3 examples/rpl-dra-ml/generate_sim.py \
    --node-count 50 \
    --attack-ratio "$ratio" \
    --seed 1 \
    --duration-ms 600000 \
    --output "examples/rpl-dra-ml/generated/rpl-dra-50n-${ratio}p-seed1.csc"
done
```

## Run Headless Cooja

```bash
cd tools/cooja
for ratio in 10 20 30; do
  ./gradlew run --args="--no-gui \
    --logdir=../../examples/rpl-dra-ml/generated/logs-50n-${ratio}p-seed1 \
    ../../examples/rpl-dra-ml/generated/rpl-dra-50n-${ratio}p-seed1.csc"
done
```

The parser in `scripts/parse_cooja_logs.py` converts the resulting
`COOJA.testlog` files into the tabular dataset used by the benchmark models.
