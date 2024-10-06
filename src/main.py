import argparse
from os import remove
import math
import speech_recognition as sr
from pydub import AudioSegment
import whisper

def transformarAudioEnTextoOpenAI(rutaArchivo, nombreFinal):
    modelo = whisper.load_model('medium')
    texto = modelo.transcribe(rutaArchivo)
    texto = texto['text']

    archivo = open(f"{nombreFinal}.txt", "a")
    archivo.write(str(texto))
    archivo.close()

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
            archivo = open(f"{nombreFinal}.txt", "a")
            archivo.write(str(texto))
            archivo.close()

        remove(f"trozo_{i}.wav")


def main():
    # Argumentos desde la línea de comandos
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI Whisper or Google Speech Recognition.")
    parser.add_argument("audio_file_path", type=str, help="Path to the audio file")
    parser.add_argument("output_file_path", type=str, help="Path to the output text file")
    parser.add_argument("model", type=str, choices=["google", "openai"], help="Model to use for transcription: 'google' or 'openai'")

    args = parser.parse_args()

    # Almacenar los valores en variables
    ruta_audio = args.audio_file_path
    archivo_salida = args.output_file_path
    modelo_seleccionado = args.model

    # Seleccionar el modelo basado en el input
    if modelo_seleccionado == "google":
        transformarAudioEnTextoGoogle(ruta_audio, archivo_salida)
    else:
        transformarAudioEnTextoOpenAI(ruta_audio, archivo_salida)



if __name__ == "__main__":
    main()
