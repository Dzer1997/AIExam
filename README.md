## Part 1: Setting up Ollama and the local LLM

### Step 1: Install Ollama

### Step 2: Be sure the Ollama is running on your computer

### Step 3: Open the terminal and run the following command(s) (They each take up 4-5 GB):
```
ollama run llama3:latest  
ollama run mistral:latest  
ollama run deepseek-r1:latest  
```

When finished, check that all models are installed with the following command:
```
ollama list
```

---

## Part 2: How to use the program with the LLM

Make sure that the Ollama local server is running, default port is 11434.

Go to `predict.py` and modify line 7 & 8:
```python
MODEL_PATH = r"Your_path\GitHub\AIExam\house_price_model.pkl"
CATEGORIES_PATH = r"Your_path\GitHub\AIExam\src\categories_seen.pkl"
```

In the `train_model.py` modify Line 17:
```python
basePath = r"Your_path\GitHub\AIExam\Data"
```

In the README file, use one of the following commands in the terminal (option 2 is only if option 1 doesn’t work)
```
LocalAI\Scripts\activate
source ./LocalAI/Scripts/activate
```

You should see something like this:

Now use the following command:
```
pip install -r requirements.txt
```

Now that we have everything set up, we are going to train the model (this takes about 45-60 minutes). Go to `train_model.py` and wait.  
Once finished, be sure that you are still in your LocalAI terminal window and run the following command:
```
streamlit run app.py or streamlit run app.py --server.address 127.0.0.1
```

Now the following window should open up

You’re finished! Feel free to explore the application by predicting the housing prices or chat with your local LLM.