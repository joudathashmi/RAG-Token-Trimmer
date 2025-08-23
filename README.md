# 🇸🇦 Saudi Government Opportunity Evaluator

A Retrieval-Augmented Generation (RAG) system that evaluates government opportunities and projects against Saudi Arabia's PDPL (Personal Data Protection Law) and Vision 2030 policies.

## 🚀 Features

- **RAG System**: Uses vector search to retrieve relevant policy sections
- **Policy Evaluation**: Evaluates projects against PDPL and Vision 2030 requirements
- **Web Interface**: Beautiful web UI for easy interaction
- **Demo Mode**: Works without API keys for testing
- **Real-time Analysis**: Provides detailed evaluation results with reasoning

## 📋 Requirements

- Python 3.8+
- OpenAI API key (optional - demo mode available)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd saudi-government-evaluator
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API key (optional):**
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

## 🎯 Usage

### Command Line Interface
```bash
# Run with default inquiry
python3 main.py

# Run with custom inquiry
python3 main.py --inquiry "Your custom inquiry here"

# Run with web interface
python3 main.py --web
```

### Web Interface
1. Start the web server:
   ```bash
   python3 main.py --web
   ```

2. Open your browser and go to: `http://localhost:8080`

3. Enter your inquiry and click "Evaluate"

## 📊 Sample Output

The system evaluates inquiries and returns:
- **Status**: Approved/Rejected
- **Reason**: Detailed explanation
- **Criteria Met**: List of satisfied requirements
- **Gaps Identified**: Areas needing improvement
- **Context References**: Which policy sections were used

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (optional)

### Model Configuration
- Default model: `gpt-4o-mini`
- Can be changed via command line arguments

## 📁 Project Structure

```
saudi-government-evaluator/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .env                # Environment variables (create this)
├── .gitignore          # Git ignore file
└── templates/          # Web interface templates
    └── index.html      # Main web page
```

## 🧪 Demo Mode

The system includes a demo mode that works without an API key:
- Uses mock responses based on inquiry content
- Perfect for testing and demonstration
- Shows the same evaluation structure as real API calls

## 🔍 How It Works

1. **Document Processing**: Loads PDPL and Vision 2030 policy texts
2. **Chunking**: Splits documents into manageable chunks
3. **Vector Search**: Uses FAISS for semantic search
4. **Retrieval**: Finds most relevant policy sections
5. **Evaluation**: LLM evaluates inquiry against retrieved context
6. **Output**: Returns structured evaluation results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built for Saudi government compliance evaluation
- Uses OpenAI's GPT models for intelligent analysis
- FAISS for efficient vector search
- Flask for web interface

---

**Note**: This tool is designed to assist with government opportunity evaluation but should not be considered as legal advice. Always consult with qualified professionals for official compliance assessments.
