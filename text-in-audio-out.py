import base64
import asyncio
import pyaudio
import speech_recognition as sr
from azure.core.credentials import AzureKeyCredential
from rtclient import (
    ResponseCreateMessage,
    RTLowLevelClient,
    ResponseCreateParams
)
import os
from dotenv import load_dotenv

load_dotenv()

# Set environment variables or edit the corresponding values here.
api_key = os.getenv("AZURE_OPENAI_API_KEY") 
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = "gpt-4o-mini-realtime-preview"

def canal_audio():
# create canal of audio
    SAMPLE_RATE = 10000 
    CHANNELS = 1        
    FORMAT = pyaudio.paInt32

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    output=True)
    return stream

def transcript_audio():
    # Initialize the speech recognizer
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak your name:")
        audio = r.listen(source)

    try:
        texto = r.recognize_google(audio, language="pt-BR")
        print("You said: " + texto)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    return texto

def handle_audio_chunk(audio_chunk):
    stream = canal_audio()
    try:
        if stream.is_active():
            stream.write(audio_chunk)
    except OSError as e:
        print("erro in write on stream: ", e)

async def text_in_audio_out(conversation):
    async with RTLowLevelClient(
        url=endpoint,
        azure_deployment=deployment,
        key_credential=AzureKeyCredential(api_key) 
    ) as client:
        await client.send(
            ResponseCreateMessage(
                response=ResponseCreateParams(
                    modalities={"audio", "text"}, 
                    instructions=f"{conversation}"
                )
            )
        )
        done = False
        while not done:
            message = await client.recv()
            match message.type:
                case "response.done":
                    done = True
                case "error":
                    done = True
                    print(message.error)
                case "response.audio_transcript.delta":
                    print(f"Received text delta: {message.delta}")
                case "response.audio.delta":
                    buffer = base64.b64decode(message.delta)
                    print(f"Received {len(buffer)} bytes of audio data.")
                    handle_audio_chunk(buffer)
                case _:
                    pass

async def main():
    while True:
        conversation = transcript_audio()
        if conversation:
            await text_in_audio_out(conversation)
        else:
            print("Please say something to start the conversation.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())