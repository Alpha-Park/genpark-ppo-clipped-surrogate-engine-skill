import sys
import json
from client import PPOClippedSurrogate

ppo = PPOClippedSurrogate()

def handle_call(name, arguments):
    if name == "compute_loss":
        old_p = arguments["old_probs"]
        new_p = arguments["new_probs"]
        adv = arguments["advantages"]
        eps = arguments.get("epsilon", 0.2)
        inst = PPOClippedSurrogate(eps)
        return {"loss": inst.compute_loss(old_p, new_p, adv)}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_call(req.get("name"), req.get("arguments", {}))
            print(json.dumps({"id": req.get("id"), "result": res}))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
