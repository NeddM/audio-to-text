import argparse
import math
import os
from os import remove
import speech_recognition as sr
from pydub import AudioSegment
import whisper
import yt_dlp
import uuid

def convertirMP3aWAV(rutaArchivo):
    if rutaArchivo.lower().endswith('.mp3'):
        audio = AudioSegment.from_mp3(rutaArchivo)
        wav_ruta = rutaArchivo.replace('.mp3', '.wav')
        audio.export(wav_ruta, format='wav')
        return wav_ruta
    return rutaArchivo

def descargarAudioDeYoutube(url):
    output_file = f"{uuid.uuid4()}.mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_file,
        'quiet': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"Error al descargar audio: {e}")
        return None
    return output_file

def transformarAudioEnTextoOpenAI(rutaArchivo, nombreFinal):
    try:
        modelo = whisper.load_model('medium')
        texto = modelo.transcribe(rutaArchivo)
        texto = texto['text']

        with open(nombreFinal, "a") as archivo:
            archivo.write(str(texto))
    finally:
        remove(rutaArchivo)

def transformarAudioEnTextoGoogle(rutaArchivo, nombreFinal):
    seg = 50
    speech = AudioSegment.from_wav(rutaArchivo)

    batch_size = seg * 1000
    duracion = speech.duration_seconds
    batches = math.ceil(duracion / seg)

    inicio = 0
    try:
        for i in range(batches):
            trozo = speech[inicio: inicio + batch_size]
            trozo_wav = f'trozo_{i}.wav'
            trozo.export(trozo_wav, format='wav')
            inicio += batch_size

            r = sr.Recognizer()

            archivo = sr.AudioFile(trozo_wav)
            with archivo as origen:
                audio = r.record(origen)
                texto = r.recognize_google(audio, language='es')
                with open(nombreFinal, "a") as archivo:
                    archivo.write(str(texto))

            remove(trozo_wav) 
    finally:
        remove(rutaArchivo)

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI Whisper or Google Speech Recognition.")
    parser.add_argument("audio_source", type=str, help="Path to the audio file or YouTube URL (or 'skip' to do nothing)")
    parser.add_argument("output_file_path", type=str, help="Path to the output text file")
    parser.add_argument("model", type=str, choices=["google", "openai"], help="Model to use for transcription: 'google' or 'openai'")

    args = parser.parse_args()

    fuente_audio = args.audio_source
    archivo_salida = args.output_file_path
    modelo_seleccionado = args.model

    if fuente_audio == "skip":
        print("Operación omitida.")
        return

    if fuente_audio.startswith("https://www.youtube.com/"):
        fuente_audio = descargarAudioDeYoutube(fuente_audio)
        if fuente_audio is None:
            print("Error al descargar el audio de YouTube.")
            return

    fuente_audio = convertirMP3aWAV(fuente_audio)

    if modelo_seleccionado == "google":
        transformarAudioEnTextoGoogle(fuente_audio, archivo_salida)
    else:
        transformarAudioEnTextoOpenAI(fuente_audio, archivo_salida)

if __name__ == "__main__":
    main()
