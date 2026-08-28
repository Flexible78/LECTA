import torch
import torchaudio
import json
import numpy as np
import gradio as gr
from pathlib import Path
from pydub import AudioSegment
from libs.utils import get_data_list, models_path
from libs.sql_db import sql_db
from libs.tts.f5_backend import F5Model, prep_audio
from config import config, AppConfig

sound_dir = Path.cwd() / "sound"
ev_path = sound_dir / 'events'
back_path = sound_dir / 'back'
f5spk = F5Model()

def save_paths(data_path_input: str, models_path_input: str):
    AppConfig.save_user_settings({
        'data_path': data_path_input,
        'models_path': models_path_input
    })
    gr.Info("Paths saved successfully to user_settings.json")
    global config
    config = AppConfig.load_user_settings()
    return gr.update(), gr.update()

def toggle_tab():
    return gr.Tabs(visible=True, selected=0)

def select_file(data, evt: gr.SelectData):
    full_path = Path(back_path) / f'{data[evt.index[0]][0]}.wav'
    if full_path.is_file():
        return gr.update(interactive=True), full_path
    else:
        return gr.update(interactive=False), None

def delete_sound(filename, evt: gr.EventData):
    target = evt.target
    if target.elem_id == 'del_ev_sound':
        ch_file = sql_db.select('list_of_snd', {'sound_type': filename})
        if len(ch_file) == 0:
            file_path = Path(ev_path) / f'{filename}.wav'
            file_path.unlink()
            gr.Warning(f'File {filename} deleted')
            return gr.update(value='', choices=sorted(get_data_list(ev_path, "*.wav")))
        else:
            gr.Warning(f'File {filename} is in use')
            return None
    if target.elem_id == 'del_back':
        path_obj = Path(filename)
        file_path = Path(back_path) / path_obj.name
        file_path.unlink()
        gr.Warning(f'File {path_obj.name} deleted')
        return gr.update(value=get_data_list(back_path, "*.wav"))

def upload_audio(dropbox):
    ab_file = Path(dropbox)
    target_file = Path(back_path) / f'{ab_file.stem}.wav'
    audio = AudioSegment.from_file(ab_file)
    if audio.channels > 1:
        audio = audio.set_channels(1)
    audio.export(target_file, format="wav")
    return gr.update(value=get_data_list(back_path, "*.wav"))

def set_tab(evt: gr.SelectData):
    return evt.index

def get_spk_data(evt: gr.SelectData):
    if evt.value is not None:
        sp_data = f5spk.speakers_data[evt.value[0]]
        audio_np = sp_data['audio'].squeeze().cpu().numpy()
        audio = (audio_np * 32767).astype(np.int16)
        sample_rate = int(sp_data['sr'].item())
        return sp_data['name'], sp_data['text'], (sample_rate, audio), evt.value[0]

def add_speaker(speaker_id, ref_name, ref_text, ref_audio, archive_path=models_path / 'speakers_data.pt'):
    speaker_id = str(speaker_id)
    speakers_data = torch.load(archive_path, map_location='cpu')
    if speaker_id in speakers_data:
        gr.Warning(f"Speaker {speaker_id} already exists! Overwriting...")
    audio, rms, sr = prep_audio(ref_audio)
    speakers_data[speaker_id] = {
        'audio': audio.cpu(),
        'rms': rms.cpu() if torch.is_tensor(rms) else torch.tensor(rms),
        'sr': torch.tensor(sr),
        'text': ref_text,
        'name': ref_name,
        'audio_len': torch.tensor(audio.shape[-1] // 256),
        'text_len': torch.tensor(len(ref_text.encode("utf-8")))
    }

    torch.save(speakers_data, archive_path)
    gr.Info(f"Speaker {speaker_id} added")
    f5spk.speakers_list()
    return 5

def del_speaker(ref_index):
    archive_path = models_path / 'speakers_data.pt'
    if not archive_path.exists():
        print(f"Error: Archive file not found at {archive_path}")
        return False, False
    speakers_data = torch.load(archive_path, map_location='cpu')
    if ref_index not in speakers_data:
        print(f"Warning: Speaker {ref_index} not found in archive")
        return False, False
    del speakers_data[ref_index]
    torch.save(speakers_data, archive_path)
    gr.Warning(f"Speaker {ref_index} deleted")
    return gr.update(samples=[[b, a] for a, b in f5spk.speakers_list()]), 5

def on_excep_select(data, evt: gr.SelectData):
    if len(data[evt.index[0]]) == 1:
        return data[evt.index[0]][0]
    return data[evt.index[0]]

def on_sound_ev_select(data, evt: gr.SelectData):
    regexp, fname = data[evt.index[0]]
    audio_path = ev_path / f'{fname}.wav'
    return regexp, fname, audio_path

def exception_action(excp, evt: gr.EventData, excp_w_a=None):
    target = evt.target
    if target.elem_id == 'add_exept':
        sql_db.upsert('cust_dict', {'word': excp, 'transcription': excp_w_a})
        gr.Info(f"Added exception: {excp}", duration=4)
    elif target.elem_id == 'del_exept':
        sql_db.delete('cust_dict', {'word': excp})
        gr.Info(f"Removed exception: {excp}", duration=4)
    return gr.update(value=sql_db.select('cust_dict', {'word': False, 'transcription': False}))

def abbr_action(abbr, evt: gr.EventData):
    target = evt.target
    if target.elem_id == 'add_abbr':
        sql_db.upsert('exc_abrs', {'abbreviation': abbr})
        gr.Info(f"Added abbreviation: {abbr}", duration=4)
    elif target.elem_id == 'del_abbr':
        sql_db.delete('exc_abrs', {'abbreviation': abbr})
        gr.Info(f"Removed abbreviation: {abbr}", duration=4)
    return gr.update(value=sql_db.select('exc_abrs', {'abbreviation': False}))

def sound_event_action(ev_name, evt: gr.EventData, se_path=''):
    target = evt.target
    if target.elem_id == 'add_ev':
        sql_db.upsert('list_of_snd', {'pattern': ev_name, 'sound_type': se_path})
        gr.Info(f"Added event: {ev_name}", duration=4)
    elif target.elem_id == 'del_ev':
        sql_db.delete('list_of_snd', {'pattern': ev_name})
        gr.Info(f"Removed event: {ev_name}", duration=4)
    return gr.update(value=sql_db.select('list_of_snd', {'pattern': False, 'sound_type': False}))


def handle_audio_upload(full_path, evt: gr.EventData):
    target = evt.target
    if full_path is None:
        return None
    f_path = Path(full_path)
    file_name = f_path.stem
    if f_path.suffix != '.wav':
        return None
    return f_path.stem


def _events_sound_choices():
    """Возвращает отсортированный список .wav файлов из sound/events/."""
    if ev_path.exists():
        return sorted([x.name for x in ev_path.iterdir() if x.suffix == ".wav"])
    return ["complete.wav"]


def _save_completion_sound(filename):
    """Сохраняет выбранный звук завершения в user_settings."""
    if filename:
        AppConfig.save_user_settings({"completion_sound": filename})
        gr.Info(f"Completion sound: {filename}")


def settings_tab(tts_state):
    tab_index = gr.State(value=0)
    with gr.Tabs() as s_tabs:
        with gr.Tab("Background music", id=40):
            with gr.Row():
                with gr.Column():
                    audio_fe = gr.Dataframe(
                        headers=['File name'],
                        value=get_data_list(back_path, "*.wav"),
                        interactive=False,
                        type='array',
                    )
                with gr.Column():
                    back_audio_input = gr.Audio(interactive=False, type='filepath', buttons=[])
                    del_butt = gr.Button("❌ Delete file", elem_id='del_back')
                    upload_back_file = gr.UploadButton(
                        "Upload background music",
                        file_count="single",
                        file_types=[".mp3", ".wav"]
                    )

            del_butt.click(
                delete_sound,
                inputs=back_audio_input,
                outputs=audio_fe
            )
            upload_back_file.upload(
                upload_audio,
                inputs=upload_back_file,
                outputs=audio_fe
            )
            audio_fe.select(
                select_file,
                inputs=audio_fe,
                outputs=[del_butt, back_audio_input]
            )

        with gr.Tab("Exceptions dictionary", id=41):
            with gr.Row():
                with gr.Column():
                    exception_words = gr.Dataframe(
                        headers=["Exception", "With stress"],
                        value=sql_db.select('cust_dict', {'word': False, 'transcription': False}),
                        interactive=False,
                        type='array',
                    )
                    with gr.Row():
                        excp = gr.Text(show_label=False, placeholder='Exception', interactive=True)
                        excp_with_accent = gr.Text(show_label=False, placeholder='With stress', interactive=True)
                    with gr.Row():
                        add_exept_butt = gr.Button("Add exception", elem_id='add_exept')
                        del_exept_butt = gr.Button("❌ Delete", elem_id='del_exept')
                with gr.Column():
                    exception_abbrs = gr.Dataframe(
                        headers=["Abbreviation"],
                        value=sql_db.select('exc_abrs', {'abbreviation': False}),
                        interactive=False,
                        type='array',
                    )
                    with gr.Row():
                        abbr = gr.Text(show_label=False, placeholder='Abbreviation')
                    with gr.Row():
                        add_abbr_butt = gr.Button("Add", elem_id='add_abbr')
                        del_abbr_butt = gr.Button("❌ Delete", elem_id='del_abbr')

            exception_words.select(
                on_excep_select,
                inputs=exception_words,
                outputs=[excp, excp_with_accent],
            )
            exception_abbrs.select(
                on_excep_select,
                inputs=exception_abbrs,
                outputs=[abbr],
            )
            add_exept_butt.click(
                exception_action,
                inputs=[excp, excp_with_accent],
                outputs=exception_words
            )
            del_exept_butt.click(
                exception_action,
                inputs=excp,
                outputs=exception_words
            )
            add_abbr_butt.click(
                abbr_action,
                inputs=abbr,
                outputs=exception_abbrs
            )
            del_abbr_butt.click(
                abbr_action,
                inputs=abbr,
                outputs=exception_abbrs
            )

        with gr.Tab("Event sounds", id=42):
            with gr.Row():
                with gr.Column(scale=3):
                    sound_events = gr.Dataframe(
                        headers=["Event description", "Sound"],
                        value=sql_db.select('list_of_snd', {'pattern': False, 'sound_type': False}),
                        interactive=False,
                        type='array',
                    )
                with gr.Column(scale=5):
                    with gr.Row():
                        ev_audio = gr.Audio(interactive=True, type="filepath", format="wav", sources=["upload", "microphone"])
                    with gr.Row():
                        se_path = gr.Dropdown(
                            show_label=False,
                            allow_custom_value=True,
                            choices=get_data_list(ev_path, "*.wav"),
                            value=get_data_list(ev_path, "*.wav")[0] if get_data_list(ev_path, "*.wav") else None,
                            interactive=True,
                        )
                        del_sound_butt = gr.Button("❌ Delete sound", elem_id='del_ev_sound')
                    with gr.Row():
                        ev_name = gr.Text(show_label=False, placeholder='Event description')
                        add_ev_butt = gr.Button("Add event", elem_id='add_ev')
                        del_ev_butt = gr.Button("❌ Delete event", elem_id='del_ev')

            sound_events.select(
                on_sound_ev_select,
                inputs=sound_events,
                outputs=[ev_name, se_path, ev_audio],
            )
            se_path.change(
                fn=lambda path: Path(ev_path) / f'{path}.wav',
                inputs=se_path,
                outputs=ev_audio
            )
            ev_audio.change(
                handle_audio_upload,
                inputs=ev_audio,
                outputs=[se_path]
            )
            add_ev_butt.click(
                sound_event_action,
                inputs=[ev_name, se_path],
                outputs=sound_events
            )
            del_ev_butt.click(
                sound_event_action,
                inputs=ev_name,
                outputs=sound_events
            )
            del_sound_butt.click(
                delete_sound,
                inputs=se_path,
                outputs=se_path
            )

        with gr.Tab("Voice samples", id=43):
            with gr.Row():
                with gr.Column(scale=1):
                    spk_list = gr.Dataset(
                        components=['text', 'text'],
                        label="Speakers",
                        headers=['#', "Name"],
                        samples_per_page=15,
                        samples=[[b, a] for a, b in f5spk.speakers_list()],
                    )
                with gr.Column(scale=3):
                    with gr.Row():
                        ref_audio = gr.Audio(interactive=True, label='Your voice sample', sources=["upload", "microphone"])
                    with gr.Row():
                        with gr.Column(scale=1):
                            spk_index = gr.Number()
                            ref_name = gr.Textbox(
                                label='Speaker name',
                                lines=1,
                            )
                        with gr.Column(scale=7):
                            ref_text = gr.Textbox(
                                label='Text in sample',
                                lines=2,
                                placeholder="Enter the text spoken in the sample",
                                interactive=True
                            )
                    with gr.Row():
                        add_spk_butt = gr.Button("Add/edit speaker")
                        del_spk_butt = gr.Button("❌ Delete speaker")

            spk_list.select(
                get_spk_data,
                outputs=[ref_name, ref_text, ref_audio, spk_index]
            )
            add_spk_butt.click(
                add_speaker,
                inputs=[spk_index, ref_name, ref_text, ref_audio],
                outputs=tts_state
            ).then(
                fn=lambda: gr.Dataset(samples=[[b, a] for a, b in f5spk.speakers_list()]),
                outputs=spk_list
            )
            del_spk_butt.click(
                del_speaker,
                inputs=spk_index,
                outputs=[spk_list,tts_state]
            )

        with gr.Tab("Storage paths", id=44):
            with gr.Row():
                data_path_box = gr.Textbox(
                    value=str(config.data_path),
                    label="Data path (data_path)",
                    placeholder="Enter a path or select a folder",
                )
            with gr.Row():
                models_path_box = gr.Textbox(
                    value=str(config.models_path),
                    label="Models path (models_path)",
                    placeholder="Enter a path or select a folder",
                )

            save_paths_btn = gr.Button("💾 Save paths")

            save_paths_btn.click(
                save_paths,
                inputs=[data_path_box, models_path_box],
                outputs=[data_path_box, models_path_box]
            )

        with gr.Tab("🔔 Completion sound", id=45):
            with gr.Row():
                completion_sound_sel = gr.Dropdown(
                    value=config.completion_sound,
                    label="Select a sound for synthesis completion notification",
                    choices=_events_sound_choices(),
                    interactive=True,
                )

            completion_sound_sel.change(
                fn=_save_completion_sound,
                inputs=completion_sound_sel,
            )
            # Refresh the list when the tab opens
            s_tabs.select(
                fn=lambda: _events_sound_choices(),
                outputs=completion_sound_sel,
            )

        with gr.Tab("🧠 F5-TTS Quality", id=46):
            gr.Markdown(
                "### F5-TTS Inference Steps\n\n"
                "Controls how many denoising steps the F5-TTS model performs.\n\n"
                "- **4–6**: ⚡ Very fast, lower quality — good for testing\n"
                "- **8–12**: ⚖️ Balanced speed/quality (recommended)\n"
                "- **16–20**: 🎧 High quality, slower\n"
                "- **24–32**: 🔬 Maximum quality, slowest\n\n"
                "This also appears as the **'Inference steps' slider** on the TTS tab."
            )
            with gr.Row():
                f5_nfe_slider = gr.Slider(
                    4, 32,
                    value=config.noise_lvl,
                    step=2,
                    label="Default inference steps (nfe_step)",
                    info="Lower = faster synthesis, higher = better audio quality",
                    interactive=True,
                )

            def _save_nfe_steps(val):
                AppConfig.save_user_settings({"noise_lvl": int(val)})
                gr.Info(f"Default F5-TTS inference steps set to {int(val)}")

            f5_nfe_slider.change(
                fn=_save_nfe_steps,
                inputs=f5_nfe_slider,
            )

        s_tabs.select(
            set_tab,
            outputs=tab_index
        )

    return s_tabs
