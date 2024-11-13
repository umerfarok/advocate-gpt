# Advocate GPT

Advocate GPT is a question-answering system that leverages Pakistani legal documents to provide answers to users' questions. It uses a combination of language models, embedding-based retrieval, and custom processing to deliver relevant responses.

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional, but recommended for better performance)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/advocate-gpt.git
    cd advocate-gpt
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your Pakistani legal PDF files in the `data/law_books/` directory.

2. Run the setup script to process the PDFs, create embeddings, and save the data:
    ```bash
    python src/main.py
    ```
    This will:
    - Extract text from the PDF files
    - Split the text into smaller chunks
    - Create sentence embeddings for the chunks
    - Save the processed data in the [vector_store](http://_vscodecontentref_/0) directory

3. Start the API server:
    ```bash
    python src/api/server.py
    ```
    The server will start running on `http://localhost:5000`.

4. Test the system by making a POST request to the `/ask` endpoint:
    ```python
    import requests

    response = requests.post('http://localhost:5000/ask', 
                            json={'question': 'What is the punishment for theft?'})
    print(response.json())
    ```
    The response will contain:
    - The generated answer
    - The relevant source documents and their relevance scores
    - The processing time

You can also check the system's health and memory status by making GET requests to the `/health` and `/memory` endpoints.

## Configuration

The system can be further customized by modifying the following parameters:
- [pdf_processor.py](http://_vscodecontentref_/1): Adjust chunk size, overlap, and text cleaning settings.
- [embeddings_manager.py](http://_vscodecontentref_/2): Change the sentence embedding model, batch size, and vector store settings.
- [qa_model.py](http://_vscodecontentref_/3): Select a different language model for question answering, adjust temperature, and more.
- [server.py](http://_vscodecontentref_/4): Modify the API endpoint behavior and response formats.

## Deployment

To deploy the Advocate GPT system, you can:
1. Build a Docker image:
    ```bash
    docker build -t advocate-gpt .
    docker run -p 5000:5000 advocate-gpt
    ```

2. Deploy to a cloud platform (e.g., AWS, GCP, Azure) using containers or serverless functions.

3. Integrate the system into a larger application or website.

## Contributing

If you find any issues or would like to contribute to the project, please feel free to open a new issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the MIT License.# advocate-gpt
