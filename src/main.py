import argparse
from os import remove
import math
import speech_recognition as sr
from pydub import AudioSegment
import whisper
import os
import yt_dlp

def convertirMP3aWAV(rutaArchivo):
    if rutaArchivo.lower().endswith('.mp3'):
        audio = AudioSegment.from_mp3(rutaArchivo)
        wav_ruta = rutaArchivo.replace('.mp3', '.wav')
        audio.export(wav_ruta, format='wav')
        return wav_ruta
    return rutaArchivo

def descargarAudioDeYoutube(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.mp3',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return 'audio.mp3'

def transformarAudioEnTextoOpenAI(rutaArchivo, nombreFinal):
    modelo = whisper.load_model('medium')
    texto = modelo.transcribe(rutaArchivo)
    texto = texto['text']

    with open(f"{nombreFinal}", "a") as archivo:
        archivo.write(str(texto))

    remove(rutaArchivo)

def transformarAudioEnTextoGoogle(rutaArchivo, nombreFinal):
    seg = 50

    speech = AudioSegment.from_wav(rutaArchivo)

    batch_size = seg * 1000
    duracion = speech.duration_seconds
    batches = math.ceil(duracion / seg)

    inicio = 0
    for i in range(batches):
        trozo = speech[inicio: inicio + batch_size]
        trozo.export(f'trozo_{i}.wav', format='wav')
        inicio += batch_size

        r = sr.Recognizer()

        archivo = sr.AudioFile(f"trozo_{i}.wav")
        with archivo as origen:
            audio = r.record(origen)
            texto = r.recognize_google(audio, language='es')
            with open(f"{nombreFinal}", "a") as archivo:
                archivo.write(str(texto))

        remove(f"trozo_{i}.wav")

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

    if fuente_audio.startswith("http"):
        fuente_audio = descargarAudioDeYoutube(fuente_audio)

    fuente_audio = convertirMP3aWAV(fuente_audio)

    if modelo_seleccionado == "google":
        transformarAudioEnTextoGoogle(fuente_audio, archivo_salida)
    else:
        transformarAudioEnTextoOpenAI(fuente_audio, archivo_salida)

if __name__ == "__main__":
    main()
