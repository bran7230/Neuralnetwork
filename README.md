# C# Neural Network Prototype

This is a C# prototype designed to explore and implement neural networks and transformer-style models from scratch. The goal is to deeply understand the inner workings of AI systems by manually building each component.

---

## 🧠 Features Implemented

### ✅ Neural Network Core
- Manual forward pass
- Softmax function
- Gradient calculation (backpropagation)
- Custom training loop with weight updates
- ReLU and Sigmoid activations

### ✅ Transformer Prototype
- Token ID input handling
- Multi-layer Transformer block structure
- Logit output with softmax sampling
- One-token generation
- In-progress: **multi-token autoregressive generation**

### 🧪 Training & Output
- Simple string-to-token system
- Model can generate basic sequences like `"I am"`
- Training runs over epochs, showing logits and token evolution

---

## 💡 Goals of the C# Version

- Learn neural networks and transformers **by hand**
- Manually implement backpropagation and attention mechanisms
- Build a functional GPT-style model from the ground up
- Use this version as a **stepping stone to a C++ implementation**

---

## 🛠️ What's Next
- Finish multi-token generation
- Add a basic tokenizer with vocabulary mapping
- Expand attention and positional encoding
- Port to C++ for better performance and low-level control

---

## 🔧 Requirements
- .NET SDK 6.0+
- Run with: `dotnet run`

---

## 👨‍💻 Author
Built by **Brandon Thomas Trumble**  
This project lays the foundation for a fully custom AI architecture, focused on learning, reasoning, and code generation.
