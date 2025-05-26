## Part 1: Setting up Ollama and the local LLM

### Step 1: Install Ollama

### Step 2: Be sure the Ollama is running on your computer
![pic1](https://github.com/user-attachments/assets/d7440262-394e-40b5-991b-d29130069520)
### Step 3: Open the terminal and run the following command(s) (They each take up 4-5 GB):
```
ollama run llama3:latest  
ollama run mistral:latest  
ollama run deepseek-r1:latest  
```

### Step 4: When finished, check that all models are installed with the following command:
```
ollama list
```
![pic2](https://github.com/user-attachments/assets/6d49d38d-01d9-4a1d-bbf0-92ea9e053d27)
---

## Part 2: How to use the program with the LLM

Make sure that the Ollama local server is running, default port is 11434.

![pic3](https://github.com/user-attachments/assets/7937b5b9-2178-418c-8ea9-7bc6dcfdef14)

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
![pic4](https://github.com/user-attachments/assets/023d9ac2-b907-4ccc-8dfa-4cddc003f2c6)

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
![pic5](https://github.com/user-attachments/assets/abc8a067-6ef1-48fb-b0fb-f0f67aa99dfb)

You’re finished! Feel free to explore the application by predicting the housing prices or chat with your local LLM.
