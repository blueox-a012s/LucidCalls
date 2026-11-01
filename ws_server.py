from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
import uvicorn
import torch
import librosa
import asyncio
import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoModelForCausalLM, AutoTokenizer
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origin_regex=".*", allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(max_workers=1)

SCAM_KEYWORDS = ["阿媽", "阿爸", "個仔", "個女", "孫", "換咗電話", "新號碼", "急需要", "急需錢", "轉錢", "保釋金", "交保釋", "唔好報警", "唔好同人講"]

# 1. Force ONNX to CPU
print("Loading ONNX Deepfake Detector (CPU)...")
ort_session = ort.InferenceSession("deepfake_detector.onnx", providers=['CPUExecutionProvider'])

# 2. Force PyTorch to CPU
device = "cpu"
print("Loading Whisper ASR (CPU)...")
asr_model_id = "alvanlii/whisper-small-cantonese"
asr_processor = WhisperProcessor.from_pretrained(asr_model_id)
asr_model = WhisperForConditionalGeneration.from_pretrained(asr_model_id).to(device)

print("Loading Qwen LLM (CPU)...")
llm_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_id).to(device)

def extract_audio_safely(input_path):
    output_path = f"clean_{uuid.uuid4().hex}.wav"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path]
    subprocess.run(cmd, check=True, capture_output=True)
    audio, _ = librosa.load(output_path, sr=16000, mono=True)
    os.remove(output_path)
    return audio

def check_keywords(text):
    return [kw for kw in SCAM_KEYWORDS if kw in text] if text else []

def analyze_audio_features(audio, sr=16000):
    try:
        f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_var = float(np.var(f0)) if len(f0) > 0 else 0.0
        intervals = librosa.effects.split(audio, top_db=30)
        speech_dur = sum([end - start for start, end in intervals]) / sr
        pause_ratio = max(0.0, 1.0 - (speech_dur / (len(audio) / sr)))
    except Exception:
        pitch_var, pause_ratio = 0.0, 0.0
    return pitch_var, pause_ratio

def process_audio_chunk(audio_data, prev_fake_prob=0.0):
    try:
        CUT_LENGTH = 64600
        audio_padded = np.pad(audio_data, (0, max(0, CUT_LENGTH - len(audio_data))), 'constant')[:CUT_LENGTH]
        
        # ONNX CPU Inference
        audio_input = np.expand_dims(audio_padded, axis=0).astype(np.float32)
        logits = ort_session.run(None, {ort_session.get_inputs()[0].name: audio_input})[0][0]
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        raw_fake_prob = float(probs[0])
        smoothed_fake_prob = (0.6 * prev_fake_prob) + (0.4 * raw_fake_prob)
        
        pitch_var, pause_ratio = analyze_audio_features(audio_padded)
        text, keywords_found = "[静音...]", []
        
        if np.max(np.abs(audio_padded)) > 0.05:
            inputs = asr_processor(audio_padded, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                ids = asr_model.generate(inputs.input_features.to(device), max_new_tokens=30)
            text = asr_processor.batch_decode(ids, skip_special_tokens=True)[0]
            keywords_found = check_keywords(text)

        return text, smoothed_fake_prob, pitch_var, pause_ratio, keywords_found
    except Exception as e:
        return "Error", prev_fake_prob, 0.0, 0.0, []

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected!")
    audio_queue = asyncio.Queue()
    current_fake_prob = 0.0
    audio_buffer = bytearray()
    CHUNK_SIZE = int(16000 * 2.0 * 2)

    async def process_loop():
        nonlocal current_fake_prob
        loop = asyncio.get_event_loop()
        while True:
            audio_np = await audio_queue.get()
            text, current_fake_prob, p_var, p_ratio, kws = await loop.run_in_executor(executor, process_audio_chunk, audio_np, current_fake_prob)
            try:
                await websocket.send_json({
                    "transcript": text, "fake_probability": round(current_fake_prob * 100, 1),
                    "pitch_variance": round(p_var, 2), "pause_ratio": round(p_ratio * 100, 1), "keywords": kws
                })
            except Exception: break
            audio_queue.task_done()

    inference_task = asyncio.create_task(process_loop())
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            if len(audio_buffer) >= CHUNK_SIZE:
                audio_np = np.frombuffer(audio_buffer[:CHUNK_SIZE], dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = audio_buffer[CHUNK_SIZE:]
                if audio_queue.qsize() < 2: await audio_queue.put(audio_np)
    except WebSocketDisconnect: pass
    finally: inference_task.cancel()

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    raw_path = f"raw_{uuid.uuid4().hex}"
    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        audio = extract_audio_safely(raw_path)
        audio, _ = librosa.effects.trim(audio, top_db=30)
        if np.max(np.abs(audio)) > 0: audio = audio / np.max(np.abs(audio))

        p_var, p_ratio = analyze_audio_features(audio)
        
        inputs = asr_processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ids = asr_model.generate(inputs.input_features.to(device), max_new_tokens=100)
        text = asr_processor.batch_decode(ids, skip_special_tokens=True)[0]
        keywords_found = check_keywords(text)

        CUT_LENGTH = 64600
        audio_padded = np.pad(audio, (0, max(0, CUT_LENGTH - len(audio))), 'constant')[:CUT_LENGTH]
        audio_input = np.expand_dims(audio_padded, axis=0).astype(np.float32)
        
        logits = ort_session.run(None, {ort_session.get_inputs()[0].name: audio_input})[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        fake_prob = float(probs[0])

        prompt = f"Analyze this call:\nTranscript: {text}\nKeywords: {keywords_found}\nAI Voice Prob: {fake_prob*100:.1f}%\nPitch Var: {p_var:.1f}\nPause Ratio: {p_ratio*100:.1f}%\n\nGive a Scam Score (0-100) and explain why."
        messages = [{"role": "user", "content": prompt}]
        llm_inputs = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_tensor = llm_tokenizer([llm_inputs], return_tensors="pt").to(device)
        outputs = llm_model.generate(**inputs_tensor, max_new_tokens=150)
        llm_reasoning = llm_tokenizer.decode(outputs[0][inputs_tensor.input_ids.shape[1]:], skip_special_tokens=True)

        return {"transcript": text, "fake_probability": round(fake_prob*100,1), "pitch_variance": round(p_var,2), "pause_ratio": round(p_ratio*100,1), "keywords": keywords_found, "llm_reasoning": llm_reasoning}
    finally:
        if os.path.exists(raw_path): os.remove(raw_path)

if __name__ == "__main__":
    # Host 0.0.0.0 allows your Windows laptop to connect via the Jetson's IP address
    uvicorn.run(app, host="0.0.0.0", port=8001)
