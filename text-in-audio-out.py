import base64
import asyncio
import pyaudio
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

# Configurações que precisam ser compatíveis com o áudio recebido (ex: sample rate, channels, formato)
SAMPLE_RATE = 16000  # exemplo, ajuste para sua configuração
CHANNELS = 1         # exemplo, mono
FORMAT = pyaudio.paInt16  # exemplo, 16-bit

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True)

# Supondo que você esteja dentro de um loop onde recebe os bytes de áudio:
def handle_audio_chunk(audio_chunk):
    try:
        if stream.is_active():
            stream.write(audio_chunk)
    except OSError as e:
        print("Erro ao escrever no stream: ", e)

async def text_in_audio_out():
    async with RTLowLevelClient(
        url=endpoint,
        azure_deployment=deployment,
        key_credential=AzureKeyCredential(api_key) 
    ) as client:
        await client.send(
            ResponseCreateMessage(
                response=ResponseCreateParams(
                    modalities={"audio", "text"}, 
                    instructions="Please assist the user."
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
    await text_in_audio_out()
    # Certifique-se de fechar o stream e terminar o PyAudio após o fim do processamento
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())