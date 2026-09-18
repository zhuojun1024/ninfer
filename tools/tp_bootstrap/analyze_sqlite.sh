#!/bin/sh
DB=/tmp/tp2_prof.sqlite
which sqlite3 || echo NO_SQLITE3
ls -la $DB 2>/dev/null
sqlite3 $DB ".tables" 2>&1 | head -5
