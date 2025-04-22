# OpenLLM-1M

OpenLLM-1M is a locally deployable chatbot application powered by a Streaming Large Language Model (LLM). It supports processing and generating responses for inputs up to **1 million tokens**, making it suitable for handling long-context conversations. This project leverages **Gradio** for the user interface and is designed to run efficiently on CPU, eliminating the need for GPU resources.

By default, it is integrated with the following models:
- **[HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)**
- **[HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)**

Additionally, OpenLLM-1M allows deployment of **single or multiple models** depending on your use case.

---

## 🚀 Features

- **Long-Context Support**: Handles inputs and outputs up to 1 million tokens.
- **CPU-Friendly**: Optimized to run on standard CPUs without requiring GPU.
- **Interactive UI**: Built with Gradio for a user-friendly web interface.
- **Streaming LLM Integration**: Utilizes streaming techniques for efficient processing of large inputs.
- **Local Deployment**: Fully functional when run locally, ensuring privacy and control.
- **Flexible Model Deployment**: Supports deployment of one or multiple models simultaneously, depending on the requirements.
- **Default Model Integration**: Pre-integrated with HuggingFaceTB/SmolLM2-135M-Instruct and HuggingFaceTB/SmolLM2-360M-Instruct models.

---

## 🛠️ Installation

### Prerequisites

 **Anaconda** For environment management. 
 **Python 3.10** Ensure compatibility with dependencies. 

### Setup Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Rahulkumar010/OpenLLM-1M.git
   cd OpenLLM-1M
   ```
 

2. **Create and Activate a Conda Environment**:

   ```bash
   conda create -n openllm python=3.10
   conda activate openllm
   ```
 

3. **Install Required Packages**:

   ```bash
   pip install uv
   uv pip install -r requirements_dev.txt
   ```
 
---

## 🧠 Running the Application

### Terminal 1: Start the Model Serve


```bash
cd FastChat
python model_server.py
```
 

### Terminal 2: Launch the Gradio Interface


```bash
python app.py
```
 

After executing these commands, open your browser and navigate to the provided local URL to interact with the chatbot. 

---

## 🔧 To Do

- [ ] **Dockerization**: Containerize the application for easier deployment across different environments. 
- [ ] **UI Enhancements**: Add functionality to enable or disable streaming through the user interface.
- [ ] **Easier Model Integration**: Simplify the process of integrating new models into the application. Currently, adding a new model requires modifying the codebase, but future updates aim to provide a more modular approach for easier integration of additional models.

---

## 📚 References & Inspiration

- [FastChat](https://github.com/lm-sys/FastChat.gt): Core model server implementaion. 
- [Streaming LLM (MIT Han Lab)](https://github.com/mit-han-lab/streaming-llm.gt): Streaming techniques for efficient LLM procesing 
- Qwen5: A reference model for long-context LMs. 

---

## 📬 Contributions

This project is a personal exploration.  However, contributions are welcome!  Feel free to fork the repository, submit issues, or propose enhancements via pull requests. 

---

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Contact

For questions or feedback, please reach out to [rahul01110100@gmail.com](mailto:rahul01110100@gmail.com).

