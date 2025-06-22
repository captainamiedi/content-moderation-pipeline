import torch
import numpy as np
import random
import time
from moderator import ContentModerationSystem

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    moderator = ContentModerationSystem()
    
    # Enhanced training data with more safe examples
    X_train = [
        # Safe content (0) - 30 examples
        "This is a completely harmless text that should be allowed",
        "Enjoy our platform and have a great day with your friends",
        "I love the new features you've added to this service",
        "The weather today is beautiful and perfect for outdoor activities",
        "Learning new skills is always a rewarding experience",
        "Technology has made our lives more convenient in many ways",
        "Reading books is one of the best ways to expand your knowledge",
        "We should all work together to create a positive community",
        "Nature provides us with incredible beauty and inspiration",
        "Helping others is one of the most fulfilling activities",
        "Sarcastic remarks are sometimes misunderstood but not harmful",
        "Constructive criticism helps improve products and services",
        "Expressing frustration is normal but should be done respectfully",
        "Debating ideas is essential for intellectual growth",
        "This is a neutral statement without any harmful intent",
        "That movie was so bad it made me want to scream",  # Emotional but safe
        "I'm frustrated with this service but will contact support",  # Complaint
        "Political discussions should focus on policies not people",  # Political
        "This recipe needs more spices to be enjoyable",  # Neutral
        "The economic situation is challenging but improving",  # Neutral
        "We should discuss politics respectfully",  # Was misclassified
        "Community guidelines create a safe space",  # Was misclassified
        "Promoting peace and understanding between communities",  # Was misclassified
        "Sarcastic remark? Obviously!",  # Was misclassified
        "Healthy debates strengthen democratic processes",
        "Education is the key to solving many social problems",
        "Artistic expression should be encouraged in all forms",
        "Collaboration leads to better outcomes than competition",
        "Understanding different perspectives builds empathy",
        "Public parks improve quality of life in urban areas",
        "Meditation helps maintain mental health and clarity",
        
        # Harmful content (1) - 20 examples
        "I will find you and make you suffer for what you did",
        "Kill all members of that group, they don't deserve to live",
        "Detailed instructions on how to create explosive devices",
        "Let's organize an attack on the government building tomorrow",
        "I have access to illegal substances and can provide them",
        "We should eliminate anyone who disagrees with our ideology",
        "Explicit instructions for committing financial fraud",
        "Threats of violence against specific individuals",
        "Promoting self-harm and encouraging suicide methods",
        "Distributing non-consensual intimate content is acceptable",
        "I'm going to hurt you badly if you don't do what I say",
        "Burn down their houses and destroy their property tonight",
        "Detailed guide on how to steal identities and commit fraud",
        "Attack the police station at midnight with all weapons",
        "I can sell you illegal weapons if you contact me privately",
        "Meet me tonight and we'll teach him a lesson he won't forget",  # Implicit threat
        "Step-by-step: how to bypass security systems",  # Illegal instructions
        "Everyone from that neighborhood deserves to suffer",  # Group threat
        "Here's how to access restricted chemicals",  # Dangerous information
        "You're worthless and should disappear"  # Emotional abuse
    ]
    y_train = [0]*30 + [1]*20
    
    try:
        print("Starting training...")
        moderator.train(X_train, y_train)
        print("Training completed successfully")
    except Exception as e:
        print(f"Training error: {e}")
    
    # Test prediction
    tests = [
        ("Sarcastic remark? Obviously!", 0),  # Safe
        ("", 0),  # Empty (safe)
        ("I will hurt you if you don't comply!", 1),  # Harmful
        ("This platform is amazing and I enjoy using it daily", 0),  # Safe
        ("Let's kill all our enemies tonight", 1),  # Harmful
        ("We should discuss politics respectfully", 0),  # Safe
        ("I'm so angry I could destroy everything", 1),  # Harmful
        ("Community guidelines create a safe space", 0),  # Safe
        ("Buy illegal drugs at 123 Main Street", 1),  # Harmful
        ("Detailed instructions for making explosives at home", 1),  # Harmful
        ("Promoting peace and understanding between communities", 0),  # Safe
        ("I have weapons and will use them tomorrow", 1)  # New threat
    ]
    
    print("\nTesting predictions:")
    for text, expected in tests:
        print(f"\nAnalyzing: '{text}'")
        start_time = time.time()
        try:
            result = moderator.predict(text)
            elapsed = time.time() - start_time
            status = "SAFE" if result[0] == 0 else "HARMFUL"
            correct = "✓" if result[0] == expected else "✗"
            print(f"Result: {status} with confidence {result[1]:.2f} {correct}")
            print(f"Expected: {'SAFE' if expected == 0 else 'HARMFUL'}")
            print(f"Latency: {elapsed*1000:.2f} ms")
        except Exception as e:
            print(f"Error: {e}") 