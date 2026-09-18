#!/usr/bin/env bash
sed -n '/^bindings/,$p' /tmp/artifact_dump.txt | head -140