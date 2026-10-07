# Sourced by action.yml's steps. apt_retry <seconds> <apt-get args...> runs
# apt-get as root under `timeout <seconds>`, up to three attempts, running
# `dpkg --configure -a` between attempts in case a killed attempt left dpkg half
# done. Acquire::Retries and the connection timeouts alone did not end a mirror
# stall (CI run 37674900895), so the hard bound is `timeout`.
archives="$HOME/.cache/apt-archives"
apt_opts=(
  -o Acquire::Retries=3
  -o Acquire::http::Timeout=30
  -o Acquire::https::Timeout=30
  -o "Dir::Cache::Archives=$archives"
  -o APT::Keep-Downloaded-Packages=true
  -o APT::Sandbox::User=root
)

apt_retry() {
  local seconds=$1 attempt
  shift
  for attempt in 1 2 3; do
    if sudo timeout "$seconds" apt-get "${apt_opts[@]}" "$@"; then
      return 0
    fi
    echo "apt-get $1 attempt $attempt failed or timed out after ${seconds}s" >&2
    if [ "$attempt" -lt 3 ]; then
      sudo dpkg --configure -a
    fi
  done
  return 1
}
