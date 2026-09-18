#!/usr/bin/env bash
# Show layer 0 (GDN), layer 1 (GDN?), and find a full-attention layer
grep -E '^text/layers/(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)\/' /tmp/bindings_dump.txt | head -60
echo '=== layer types pattern: count attention vs gdn layers ==='
grep -c 'attention/query' /tmp/bindings_dump.txt
grep -c 'gdn/query' /tmp/bindings_dump.txt