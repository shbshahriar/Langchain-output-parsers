# LangChain Output Parsers Practice Project

This project demonstrates the use of LangChain's output parsers with Hugging Face models to handle structured outputs such as JSON, Pydantic objects, and more. It includes examples of generating structured data, detailed reports, and summaries using various LangChain components.

## Features

- **Structured Output Parsing**: Generate and parse structured outputs like JSON and Pydantic objects.
- **Custom Prompt Templates**: Use prompt templates to customize model behavior.
- **Hugging Face Integration**: Leverage Hugging Face models for text generation tasks.
- **Environment Variable Management**: Use `.env` files to manage API keys and configurations.

## File Architecture

```
Langchain-output-parsers/
├── .env                          # Environment variables for API keys
├── HF_login.py                   # Script to log in to Hugging Face
├── jsonoutputparser(manully).py  # Example of manually parsing JSON output
├── jsonoutputparser_chain.py     # Example of using JsonOutputParser with LangChain
├── pydanticoutputparser.py       # Example of using PydanticOutputParser
├── requirements.txt              # Required Python dependencies
├── stroutputparser.py            # Example of generating reports and summaries
├── stroutputparser_chain.py      # Example of chaining prompts with StrOutputParser
├── structuredoutputparser.py     # Example of using StructuredOutputParser
└── README.md                     # Project documentation (this file)
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Langchain-output-parsers
   ```

2. **Install Dependencies**:
   Use the `requirements.txt` file to install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the project directory (if not already present) and add your API keys:
   ```env
   OPENAI_API_KEY = "your-openai-api-key"
   ANTHROPIC_API_KEY = "your-anthropic-api-key"
   GOOGLE_API_KEY = "your-google-api-key"
   HUGGINGFACE_API_KEY = "your-huggingface-api-key"
   ```

4. **Run the Scripts**:
   Execute any of the Python scripts to see the output. For example:
   ```bash
   python structuredoutputparser.py
   ```

## Usage Examples

### 1. **Structured Output Parsing**
   - File: `structuredoutputparser.py`
   - Generates structured outputs (e.g., facts about a topic) using `StructuredOutputParser`.

### 2. **Pydantic Output Parsing**
   - File: `pydanticoutputparser.py`
   - Generates structured data using Pydantic models.

### 3. **JSON Output Parsing**
   - File: `jsonoutputparser_chain.py`
   - Parses JSON outputs using `JsonOutputParser`.

### 4. **Report and Summary Generation**
   - File: `stroutputparser_chain.py`
   - Generates detailed reports and concise summaries using chained prompts.


## License

This project is for educational purposes and does not include any specific license. Feel free to modify and use it as needed.
