import runpod, os

# Load API key from environment or .env file
api_key = os.environ.get("RUNPOD_API_KEY")
if not api_key:
    env_path = os.path.expanduser("~/.openclaw/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("RUNPOD_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                    break

runpod.api_key = api_key
pod = runpod.get_pod('iqlqpq5pj0dfgw')
rt = pod.get('runtime', {})
for p in rt.get('ports', []):
    print(f"{p['privatePort']} -> {p['ip']}:{p['publicPort']} (public: {p['isIpPublic']})")
