import torch
from TTS.api import TTS
import gradio as gr
from rvc import Config, load_hubert, get_vc, rvc_infer
import gc, os, sys, argparse, requests
import gdown
from pathlib import Path

parser = argparse.ArgumentParser(
    prog='XTTS-RVC-UI',
    description='Gradio UI para XTTSv2 y RVC'
)

parser.add_argument('-s', '--silent', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

if args.silent: 
    print('Activando modo silencioso.')
    sys.stdout = open(os.devnull, 'w')

def download_models():
    # Descarga de archivos RVC
    rvc_files = ['hubert_base.pt', 'rmvpe.pt']

    for file in rvc_files: 
        if not os.path.isfile(f'./models/{file}'):
            print(f'Descargando {file}')
            r = requests.get(f'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/{file}')
            with open(f'./models/{file}', 'wb') as f:
                f.write(r.content)

    # Descarga de un modelo RVC específico desde Google Drive
    rvc_model_url = "https://drive.google.com/uc?id=10SXLxWd2_wR3N4pJEENSsD3mbuAY__GR"
    rvc_model_destination = "./rvcs/Pedro_RVC.pth"
    gdown.download(rvc_model_url, rvc_model_destination, quiet=False)

    # Descarga de archivos XTTS desde Google Drive
    folder_url = "https://drive.google.com/drive/folders/1h-7Peta7OU4q3egdgpZI7jNh0QrWxNhw?usp=sharing"
    folder_id = folder_url.split('/')[-1]
    destination_path = "./models/xtts"
    os.makedirs(destination_path, exist_ok=True)
    gdown.download_folder(f"https://drive.google.com/drive/folders/{folder_id}", output=destination_path, quiet=False, use_cookies=False)

    # Descarga de audios desde Google Drive
    voices_folder_url = "https://drive.google.com/drive/folders/1wxFqSxYqHlBCnEG7O7_NDUtxBgfhdRTV?usp=drive_link"
    voices_folder_id = voices_folder_url.split('/')[-1]
    voices_destination_path = "./voices"
    os.makedirs(voices_destination_path, exist_ok=True)
    gdown.download_folder(f"https://drive.google.com/drive/folders/{voices_folder_id}", output=voices_destination_path, quiet=False, use_cookies=False)

[Path(_dir).mkdir(parents=True, exist_ok=True) for _dir in ['./models/xtts', './voices', './rvcs']]

download_models()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Dispositivo: " + device)

config = Config(device, device != 'cpu')
hubert_model = load_hubert(device, config.is_half, "./models/hubert_base.pt")
tts = TTS(model_path="./models/xtts", config_path='./models/xtts/config.json').to(device)
voices = []
rvcs = []
default_lang = "es"

def get_rvc_voices():
    global voices 
    voices = os.listdir("./voices")
    print('Lista de voces y RVC actualizada!')
    return gr.update(choices=voices, value=voices[0] if len(voices) > 0 else '')

def infer_voice(voice, pitch_change):
    modelname = os.path.splitext(voice)[0]
    print("Usando modelo RVC: " + modelname)
    rvc_model_path = "./rvcs/Pedro_RVC.pth"
    rvc_index_path = ""

    rvc_data.load_cpt(modelname, rvc_model_path)
    
    rvc_infer(
        index_path=rvc_index_path, 
        index_rate=0.5,  # Fijar tasa de índice a un valor por defecto
        input_path=voice, 
        output_path="./outputrvc.wav", 
        pitch_change=pitch_change, 
        f0_method="rmvpe", 
        cpt=rvc_data.cpt, 
        version=rvc_data.version, 
        net_g=rvc_data.net_g, 
        filter_radius=3, 
        tgt_sr=rvc_data.tgt_sr, 
        rms_mix_rate=0.25, 
        protect=0, 
        crepe_hop_length=0, 
        vc=rvc_data.vc, 
        hubert_model=hubert_model
    )
    return "./outputrvc.wav"

audio_counter = 1

def save_audio(audio):
    global audio_counter
    filename = f"./voices/audio_personalizado_nro_{audio_counter:02d}.wav"
    audio_counter += 1
    with open(filename, "wb") as f:
        f.write(audio)
    return filename

# Interfaz de Gradio
with gr.Blocks() as interface:
    gr.Markdown("# Pedro Labattaglia TTS")
    gr.Markdown("### Elige un audio de referencia que influye en la prosa y emocionalidad del habla generada")
    
    with gr.Row():
        with gr.Column():
            voice_dropdown = gr.Dropdown(label="Selecciona un audio de referencia de Pedro", choices=voices, value=voices[0] if len(voices) > 0 else '')
            
            pitch_slider = gr.Slider(minimum=-12, maximum=12, step=1, label="Cambio de tono", value=0)
            
            audio_input = gr.Audio(source="upload", type="filepath", label="Sube tu propio audio de referencia (recomendado hasta 30 segundos)")
            audio_recorder = gr.Audio(source="microphone", type="filepath", label="Graba tu propio audio de referencia")

        with gr.Column():
            audio_output = gr.Audio(label="Audio convertido", elem_id="audio_output", type="filepath")

    infer_button = gr.Button("Convertir voz")
    infer_button.click(fn=infer_voice, inputs=[voice_dropdown, pitch_slider], outputs=audio_output)
    audio_recorder.change(fn=save_audio, inputs=audio_recorder, outputs=voice_dropdown)

interface.launch(server_name="0.0.0.0", server_port=5000, quiet=True, share=True)

# delete later

class RVC_Data:
    def __init__(self):
        self.current_model = {}
        self.cpt = {}
        self.version = {}
        self.net_g = {} 
        self.tgt_sr = {}
        self.vc = {} 

    def load_cpt(self, modelname, rvc_model_path):
        if self.current_model != modelname:
            print("Cargando nuevo modelo")
            del self.cpt, self.version, self.net_g, self.tgt_sr, self.vc
            self.cpt, self.version, self.net_g, self.tgt_sr, self.vc = get_vc(device, config.is_half, config, rvc_model_path)
            self.current_model = modelname

rvc_data = RVC_Data()

if __name__ == "__main__":
    main()
