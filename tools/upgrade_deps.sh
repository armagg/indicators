#!/bin/bash
# Indicators: upgrade one package at a time, full suite after each.
cd "$(dirname "$0")/.." || exit 1
PY=venv/bin/python

step () {
  local pkg="$1" want="$2"
  printf '=== %-12s -> %-9s ' "$pkg" "$want"
  venv/bin/pip install -q "${pkg}==${want}" 2>&1 | grep -viE '^\s*$|notice|To update' | head -2
  local have
  have=$(venv/bin/pip show "$pkg" 2>/dev/null | awk '/^Version:/{print $2}')
  if [ "$have" != "$want" ]; then
    printf '| INSTALL FAILED (have=%s)\n' "$have"; return 1
  fi
  local out
  out=$($PY -m pytest -q -p no:cacheprovider 2>&1 | tail -1)
  printf '| %s\n' "$out"
  echo "$out" | grep -qE '^[0-9]+ passed' || { echo "  TESTS FAILED"; return 1; }
  echo "$out" | grep -qE 'failed|error' && { echo "  TESTS FAILED"; return 1; }
  return 0
}

# pandas 3 is the whole point of this batch: it goes first, with the
# datetime-resolution fix already in place, so any fallout is attributable.
step pandas      3.0.6  || exit 1
step numpy       2.5.4  || exit 1
step Flask       3.1.3  || exit 1
step Werkzeug    3.1.9  || exit 1
step MarkupSafe  3.0.4  || exit 1
step click       8.5.0  || exit 1
step pymongo     4.19.0 || exit 1
step dnspython   2.9.0  || exit 1
step pytz        2026.5 || exit 1
step tzdata      2026.5 || exit 1
step iniconfig   2.3.1  || exit 1
step pluggy      1.7.0  || exit 1
step packaging   26.3   || exit 1
step Pygments    2.21.0 || exit 1

printf '\n########## ALL STEPS GREEN ##########\n'
