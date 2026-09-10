from client import PPOClippedSurrogate

def main():
    print("=== Testing PPO Clipped Surrogate Engine ===")
    ppo = PPOClippedSurrogate(epsilon=0.2)

    old_probs = [0.5, 0.4, 0.8]
    new_probs = [0.55, 0.6, 0.75]
    advantages = [1.2, -0.5, 0.3]

    loss = ppo.compute_loss(old_probs, new_probs, advantages)
    print("Calculated PPO Clipped Objective Loss:", round(loss, 4))
    assert loss != 0.0
    print("=== All tests passed successfully! ===")

if __name__ == "__main__":
    main()
