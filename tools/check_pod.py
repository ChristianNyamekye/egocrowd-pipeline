import runpod, sys, os, time
runpod.api_key = open(os.path.expanduser("~/.openclaw/.env")).read().split("RUNPOD_API_KEY=")[1].split("\n")[0].strip()
pod_id = sys.argv[1] if len(sys.argv) > 1 else "7vyo5b1bjik513"
for i in range(20):
    p = runpod.get_pod(pod_id)
    rt = p.get("runtime") or {}
    ports = rt.get("ports", [])
    ssh = [x for x in ports if x.get("privatePort") == 22]
    if ssh:
        print(f"UP {ssh[0]['ip']} {ssh[0]['publicPort']}")
        sys.exit(0)
    print(f"{(i+1)*15}s: BOOTING")
    time.sleep(15)
print("TIMEOUT")
sys.exit(1)
