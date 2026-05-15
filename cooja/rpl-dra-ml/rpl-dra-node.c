#include "contiki.h"
#include "dev/moteid.h"
#include "lib/random.h"
#include "net/link-stats.h"
#include "net/ipv6/simple-udp.h"
#include "net/ipv6/uip-debug.h"
#include "net/routing/rpl-lite/rpl.h"
#include "net/routing/routing.h"
#include "sys/node-id.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UDP_PORT 1234
#define SEND_INTERVAL (20 * CLOCK_SECOND)
#define TRACE_INTERVAL (10 * CLOCK_SECOND)

#ifdef ROOT_NODE
#define ROLE_NAME "root"
#else
#ifdef DRA_ATTACK
#define ROLE_NAME "attacker"
#else
#define ROLE_NAME "benign"
#endif
#endif

static struct simple_udp_connection udp_conn;
static uint16_t seqno;

PROCESS(rpl_dra_process, "RPL DRA trace node");
AUTOSTART_PROCESSES(&rpl_dra_process);

static uint16_t
local_id(void)
{
  return node_id != 0 ? node_id : (uint16_t)simMoteID;
}

static uint16_t
neighbor_id(const linkaddr_t *lladdr)
{
  if(lladdr == NULL) {
    return 0;
  }
#if LINKADDR_SIZE >= 2
  return ((uint16_t)lladdr->u8[LINKADDR_SIZE - 2] << 8)
    | lladdr->u8[LINKADDR_SIZE - 1];
#else
  return lladdr->u8[0];
#endif
}

#ifndef ROOT_NODE
static void
set_global_address(void)
{
  uip_ipaddr_t ipaddr;
  const uip_ipaddr_t *default_prefix = uip_ds6_default_prefix();

  uip_ip6addr_copy(&ipaddr, default_prefix);
  uip_ds6_set_addr_iid(&ipaddr, &uip_lladdr);
  uip_ds6_addr_add(&ipaddr, 0, ADDR_AUTOCONF);
}
#endif

static void
print_trace(const char *event, uint16_t seq, unsigned long delay_ticks)
{
  rpl_instance_t *instance = rpl_get_default_instance();
  rpl_nbr_t *parent = NULL;
  const struct link_stats *stats = NULL;
  const linkaddr_t *parent_lladdr = NULL;
  uint16_t rank = RPL_INFINITE_RANK;
  uint16_t dag_rank = 0;
  uint16_t parent_rank = RPL_INFINITE_RANK;
  uint16_t parent_metric = 0xffff;
  uint16_t etx = 0xffff;
  int16_t rssi = LINK_STATS_RSSI_UNKNOWN;
  uint16_t nbr_count = 0;

  if(instance != NULL && instance->used) {
    rank = instance->dag.rank;
    dag_rank = rank == RPL_INFINITE_RANK ? 0 : DAG_RANK(rank);
    parent = instance->dag.preferred_parent;
    nbr_count = rpl_neighbor_count();
  }

  if(parent != NULL) {
    parent_rank = parent->rank;
    parent_metric = rpl_neighbor_get_link_metric(parent);
    parent_lladdr = rpl_neighbor_get_lladdr(parent);
    stats = rpl_neighbor_get_link_stats(parent);
  }

  if(stats != NULL) {
    etx = stats->etx;
    rssi = stats->rssi;
  }

  printf("TRACE event=%s time_ticks=%lu node=%u role=%s attack=%u seq=%u "
         "rank=%u dag_rank=%u parent=%u parent_rank=%u parent_metric=%u "
         "etx=%u rssi=%d nbrs=%u delay_ticks=%lu\n",
         event,
         (unsigned long)clock_time(),
         local_id(),
         ROLE_NAME,
#ifdef DRA_ATTACK
         1,
#else
         0,
#endif
         seq,
         rank,
         dag_rank,
         neighbor_id(parent_lladdr),
         parent_rank,
         parent_metric,
         etx,
         rssi,
         nbr_count,
         delay_ticks);
}

static void
udp_rx(struct simple_udp_connection *c,
       const uip_ipaddr_t *sender_addr,
       uint16_t sender_port,
       const uip_ipaddr_t *receiver_addr,
       uint16_t receiver_port,
       const uint8_t *data,
       uint16_t datalen)
{
  uint16_t sender_id;
  uint16_t seq;
  unsigned long sent_ticks;
  unsigned long delay_ticks = 0;

  if(sscanf((const char *)data, "%hu,%hu,%lu", &sender_id, &seq, &sent_ticks) == 3) {
    delay_ticks = (unsigned long)clock_time() - sent_ticks;
    printf("RX time_ticks=%lu root=%u sender=%u seq=%u delay_ticks=%lu bytes=%u\n",
           (unsigned long)clock_time(),
           local_id(),
           sender_id,
           seq,
           delay_ticks,
           datalen);
    print_trace("rx", seq, delay_ticks);
  } else {
    printf("RX_UNPARSED time_ticks=%lu node=%u bytes=%u\n",
           (unsigned long)clock_time(),
           local_id(),
           datalen);
  }

  (void)c;
  (void)sender_addr;
  (void)sender_port;
  (void)receiver_addr;
  (void)receiver_port;
}

PROCESS_THREAD(rpl_dra_process, ev, data)
{
  static struct etimer trace_timer;
#ifndef ROOT_NODE
  static struct etimer send_timer;
#endif
#ifndef ROOT_NODE
  uip_ipaddr_t root_ipaddr;
#endif

  PROCESS_BEGIN();

#ifdef ROOT_NODE
  NETSTACK_ROUTING.root_start();
#else
  set_global_address();
#endif

  simple_udp_register(&udp_conn, UDP_PORT, NULL, UDP_PORT, udp_rx);

  etimer_set(&trace_timer, TRACE_INTERVAL);
#ifndef ROOT_NODE
  etimer_set(&send_timer, SEND_INTERVAL + (random_rand() % SEND_INTERVAL));
#endif

  print_trace("boot", 0, 0);

  while(1) {
    PROCESS_WAIT_EVENT();

    if(etimer_expired(&trace_timer)) {
      print_trace("periodic", seqno, 0);
      etimer_reset(&trace_timer);
    }

#ifndef ROOT_NODE
    if(etimer_expired(&send_timer)) {
      char payload[40];
      int len;

      if(NETSTACK_ROUTING.node_is_reachable()
         && NETSTACK_ROUTING.get_root_ipaddr(&root_ipaddr)) {
        len = snprintf(payload, sizeof(payload), "%u,%u,%lu",
                       local_id(), seqno, (unsigned long)clock_time());
        printf("TX time_ticks=%lu node=%u seq=%u bytes=%d\n",
               (unsigned long)clock_time(), local_id(), seqno, len);
        simple_udp_sendto(&udp_conn, payload, len + 1, &root_ipaddr);
        print_trace("tx", seqno, 0);
        seqno++;
      } else {
        print_trace("not_reachable", seqno, 0);
      }

      etimer_set(&send_timer, SEND_INTERVAL + (random_rand() % SEND_INTERVAL));
    }
#endif
  }

  PROCESS_END();
}
