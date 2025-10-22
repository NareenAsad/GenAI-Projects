# 🤖 AI Chat Assistant - Local LLM Interface

A beautiful, modern chat interface for interacting with locally installed Large Language Models (LLMs) using Ollama and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![Ollama](https://img.shields.io/badge/Ollama-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎨 **Modern Glass-morphism UI** - Beautiful gradient background with frosted glass effects
- 💬 **Real-time Chat Interface** - Smooth, responsive chat experience
- 🤖 **Local LLM Support** - Run AI models completely offline via Ollama
- 📜 **Conversation History** - Keep track of all your messages
- 🎯 **Simple & Clean** - Minimalist design, easy to use
- ⚡ **Fast & Lightweight** - Optimized performance
- 🔒 **Privacy First** - All data stays on your machine

## 📸 Screenshots

```
┌─────────────────────────────────────────┐
│  🤖 AI Chat Assistant                   │
│  🦙 Powered by Ollama - Using: llama2  │
├─────────────────────────────────────────┤
│                                         │
│  👤 User: Hello, how are you?          │
│                                         │
│  🤖 Bot: I'm doing well, thanks!       │
│      How can I help you today?         │
│                                         │
└─────────────────────────────────────────┘
│  [Type your message here...]  [Send]   │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Ollama installed on your system

### Installation

1. **Clone or download this repository**
   ```bash
   mkdir streamlit_llm_app
   cd streamlit_llm_app
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**
   - Download from: [https://ollama.ai](https://ollama.ai)
   - Follow installation instructions for your OS

5. **Pull the LLM model**
   ```bash
   ollama pull llama2
   ```

6. **Run the application**
   ```bash
   # Start Ollama server (in one terminal)
   ollama serve

   # Start Streamlit app (in another terminal)
   streamlit run app.py
   ```

7. **Open your browser**
   - Navigate to: `http://localhost:8501`

## 📁 Project Structure

```
streamlit_llm_app/
│
├── app.py                 # Main application file
├── style.css              # Custom styling
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── venv/                 # Virtual environment (not committed)
```

## 🛠️ Configuration

### Change the Default Model

Edit `app.py` and modify this line:

```python
DEFAULT_MODEL = 'llama2'  # Change to your preferred model
```

Available models you can use:
- `llama2` - General purpose (7B)
- `mistral` - Fast and efficient
- `codellama` - For coding tasks
- `phi` - Lightweight model
- `gemma` - Google's model

Pull any model with:
```bash
ollama pull <model-name>
```

## 📦 Dependencies

```
streamlit>=1.31.0
requests>=2.31.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

1. **Start a conversation**
   - Type your message in the input box at the bottom
   - Press Enter or click "Send"

2. **Clear chat history**
   - Click the "Clear Chat" button in the top-right corner

3. **View responses**
   - User messages appear with a 👤 avatar
   - Bot responses appear with a 🤖 avatar

## 🔧 Troubleshooting

### "Connection Error: Make sure Ollama is running"

**Solution:** Start the Ollama server
```bash
ollama serve
```

### "Model not found"

**Solution:** Pull the model first
```bash
ollama pull llama2
```

### "CSS file not found"

**Solution:** Ensure `style.css` is in the same directory as `app.py`

### Port already in use

**Solution:** Run on a different port
```bash
streamlit run app.py --server.port 8502
```

### Slow responses

**Solution:** Try a smaller model
```bash
ollama pull phi
# Then change DEFAULT_MODEL to 'phi' in app.py
```

## 🌟 Features in Detail

### Chat Interface
- Clean, modern design with glass-morphism effects
- Smooth animations and transitions
- Auto-scrolling chat history
- Fixed input area at bottom

### Model Integration
- Communicates with Ollama via REST API
- Supports all Ollama-compatible models
- Handles connection errors gracefully
- Timeout protection (120 seconds)

### User Experience
- Form-based input (press Enter to send)
- Loading spinner during processing
- Clear error messages
- Conversation persistence during session

## 🎨 Customization Guide

### Change Colors

**Purple Theme (Default):**
```css
background: linear-gradient(to bottom right, #1a1a2e, #16213e, #0f3460);
```

**Blue Theme:**
```css
background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
```

**Green Theme:**
```css
background: linear-gradient(to bottom right, #134e5e, #71b280);
```

### Modify Layout

Change the chat container height in `style.css`:
```css
.chat-container {
    max-height: 600px;  /* Adjust this value */
}
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - For the amazing web framework
- [Ollama](https://ollama.ai/) - For making local LLMs accessible
- [Meta AI](https://ai.meta.com/) - For the Llama 2 model

## 📧 Contact

Have questions or suggestions? Feel free to:
- Open an issue
- Submit a pull request
- Contact the maintainer

## 🔮 Future Enhancements

- [ ] Multiple model support with dropdown selector
- [ ] Export chat history to file
- [ ] Streaming responses (token-by-token)
- [ ] Voice input support
- [ ] Dark/Light theme toggle
- [ ] Custom system prompts
- [ ] Chat history persistence (save to database)
- [ ] Multi-language support

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Llama 2 Model Card](https://github.com/facebookresearch/llama)

---

**Made with ❤️ using Streamlit and Ollama**

*Star ⭐ this repository if you find it helpful!*