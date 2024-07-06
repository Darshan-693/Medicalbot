from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import os
import numpy as np
import cv2
#from tensorflow import keras
from keras.models import model_from_json
import random

DB_FAISS_PATH = 'vectorstore/db_faiss'
UPLOAD_DIR = 'uploaded_images'  # Directory to save uploaded images

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        config = {'context_length' : 2048},
        temperature = 0.5
    )
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Chainlit code for handling chat start
@cl.on_chat_start
async def start():

    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medi-Max. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)



    files = await cl.AskFileMessage(
        content="Please upload a text file to begin!", accept=["image/jpeg"]
    ).send()
    #print(files[0].path)
    all_items = os.listdir('.')
    i = len(files)-1
    while files[0].path[i] != "\\":
        i-=1
    photu = files[0].path[i+1:]
    print(photu)
    if '.files' in all_items and os.path.isdir('.files'):
        # Change the current directory to '.files'
        os.chdir('.files')
        # List contents of '.files'
        #print(os.listdir('.'))

    text_file = os.listdir('.')[0].replace("\\","/")
    print(text_file)

    json_file = open("../heartDisease.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("../heartDisease.h5")
    label = ['Myocardial Infarction','History of MI ','abnormal heartbeat','Normal']
    MI = ['''A Myocardial Infarction, or heart attack, is when something gets stuck
        in the little tubes that bring blood to your heart, like a tiny jam in a water pipe. When this happens, your heart can’t get the oxygen it needs to work properly, and that part of the heart starts to hurt and get damaged.
        It's really important to get help 
        right away so doctors can unblock the tubes and help the heart feel better.''',
        '''Myocardial Infarction (MI), commonly known as a heart attack, 
        is a medical emergency that occurs when blood flow to a part of the heart muscle (myocardium) 
        is abruptly reduced or stopped. This usually happens due to the occlusion of one or more of the coronary 
        arteries by a thrombus (blood clot) that forms on a ruptured or eroded atherosclerotic plaque. The resulting 
        ischemia leads to myocardial cell death if blood flow is not quickly restored. Key clinical markers include 
        elevated cardiac biomarkers
            such as troponins, and typical symptoms can include chest pain, shortness of breath, and diaphoresis.''',
            '''Myocardial Infarction, often called a heart attack, happens when the blood flow to a part of your heart is blocked. 
            This blockage prevents oxygen from getting to the heart muscle, causing damage. It's like a 
            "plumbing problem" in your heart's pipes, where something is clogging the flow. Common signs
            include severe chest pain, sweating, and feeling very weak. Quick treatment is crucial to minimize 
            heart damage.'''
        ]
    ABN=['''An abnormal heartbeat, also called an arrhythmia, is when your heart doesn't beat quite right. Imagine your heart is like a drum that keeps a steady rhythm, but sometimes it beats too fast, too slow, or skips a beat. This can make you feel funny, like your heart is racing or fluttering. Sometimes you need to see a doctor to help your heart get back to its regular beat.''',
        '''An abnormal heartbeat, or arrhythmia, occurs when the heart's usual rhythm is disrupted. This can mean your heart beats too fast, too slow, or irregularly. It’s like your heart’s internal clock is out of sync. Many people experience palpitations, dizziness, or shortness of breath when this happens. Arrhythmias can be harmless or a sign of a more serious issue, so it’s important to get checked by a doctor if you notice these symptoms.''',
        '''An abnormal heartbeat, or cardiac arrhythmia, refers to any deviation from the normal sinus rhythm of the heart. Arrhythmias can present as tachycardia, bradycardia, or irregular patterns, including atrial fibrillation, ventricular tachycardia, or premature contractions. They arise from various etiologies, including electrolyte imbalances, myocardial ischemia, or conduction system disorders. Symptoms can range from asymptomatic to palpitations, syncope, or hemodynamic instability. Diagnosis often involves ECG analysis, and management may include pharmacologic therapy, electrical cardioversion, or interventional procedures like ablation.''',
        '''Abnormal heartbeats, known as arrhythmias, occur when the electrical impulses that regulate heartbeats are disrupted. These can manifest as tachycardia (fast heart rate), bradycardia (slow heart rate), or irregular rhythms. Arrhythmias can result from issues in the heart's conduction system, influenced by factors such as electrolyte imbalances, ischemic heart disease, or structural changes in the myocardium. Clinically, they may present with palpitations, fatigue, or more severe symptoms depending on the underlying cause and the heart's functional status.''']
    HMI=['''The history of a heart attack, or myocardial infarction (MI), is when someone's heart had a problem in the past. It's like a story about how their heart got very sick, but doctors helped them get better so their heart could keep working.''',
        '''The history of a heart attack, or myocardial infarction (MI), refers to someone having had a serious problem with their heart in the past. It means their heart didn't get enough blood and oxygen, which caused damage. People who have had an MI often need to take special care of their heart to stay healthy.''',
        '''The history of myocardial infarction (MI) refers to a documented instance where a patient experienced significant myocardial damage due to ischemia, often resulting from coronary artery obstruction. This historical event is crucial for assessing cardiac health risks and guiding ongoing management strategies, including lifestyle modifications and medical therapies.''',
        '''The history of myocardial infarction (MI) encompasses past episodes where severe ischemia led to myocardial cell death. This often results from coronary artery disease and can cause lasting damage to the heart muscle. Understanding a patient's MI history is essential for evaluating their cardiac health and determining appropriate treatment strategies.''']
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    def predict_image(image_path, model, labels):
        image = preprocess_image(image_path)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return labels[predicted_class]

    print(f"./{text_file}/{photu}")
    image = cl.Image(path=f"./{text_file}/{photu}", name="image1", display="inline")
    disease = predict_image(f"./{text_file}/{photu}",model, label)
    if(disease == 'Myocardial Infarction'):
        await cl.Message(content="Seems like you have a condition of Myocardial Infraction.\n\n"+MI[random.randint(0,2)],elements=[image]).send()
    elif(disease == 'Normal'):
        await cl.Message(content="Your heart rate appears to be within normal range. That's great news! Keeping track of your heart health is important. ",elements=[image]).send()
    elif(disease == 'History of MI '):
        await cl.Message(content="It appears that you have a history of Myocardial Infarction (heart attack). Understanding your medical history, including any previous heart events, is important for managing your overall health. It's advisable to continue following your doctor's recommendations, such as taking prescribed medications, making lifestyle adjustments, and attending regular check-ups. This proactive approach can help reduce the risk of future complications and support your ongoing well-being.\n\n"
+HMI[random.randint(0,3)],elements=[image]).send()

    elif(disease == 'abnormal heartbeat'):
        await cl.Message(content="It seems like your heartbeat is showing some irregularities. Many people experience this at times, and it's often manageable with the right care. It's important to let your healthcare provider know about these symptoms so they can assess and guide you on the best steps forward. Remember, there are effective treatments available, and early detection can make a big difference in managing your heart health.\n\n"+ABN[random.randint(0,3)],elements=[image]).send()
    os.remove(f"{text_file}")
    os.chdir('../')
    
    #Let the user know that the system is ready
   
@cl.on_message
async def main(message: cl.Message):
    
        
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=False, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += "\nNo sources found"
    await cl.Message(content=answer).send()


    

    