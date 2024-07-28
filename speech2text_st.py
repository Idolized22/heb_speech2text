import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# from audiorecorder import audiorecorder
from st_audiorec import st_audiorec
import yaml

class Speech2Text_LLM:
    
    def __init__(self, api_key, instructions = '') -> None:
        # self.load_api_key(api_key)
        self.init_gemini(api_key, instructions)
        
    def get_instructions(self):
        return self.instructions
       
    def init_gemini(self, api_key, instructions: str):
        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 0.25,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192*16,
            "response_mime_type": "text/plain",
            }
        
        if instructions:
            self.instructions = instructions
        else:
            self.instructions = ' המשימה שלך היא לבצע דיבור לטקסט, תתנהג כמו מזכירה ותמיר את ההקלטה שהמשתמש מספק לך לטקסט.בצע את ההמרה מילה במילה. שמור על הקשר הגיוני בין המילים. חשוב שלב אחרי שלב.'
    
        self.llm_model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",#"gemini-1.5-pro", #"gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=self.instructions,
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_DEROGATORY: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_TOXICITY: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_MEDICAL: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_DANGEROUS: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }        
        )
        self.chat_session = self.llm_model.start_chat()
        
    def upload_to_gemini(self, path, mime_type=None):
        """Uploads the given file to Gemini.

        See https://ai.google.dev/gemini-api/docs/prompting_with_media
        """
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file 
      
    # def get_session_history(self, file_path):
    #     session_history = []
    #     files = [self.upload_to_gemini(file_path, mime_type="audio/ogg")]
    #     session_history = [
    #         {"role": "user",
    #          "parts": [files[0]],
    #          },
    #         ]       
    #     return session_history
            
             
    def apply(self, path_audio2process: str):
          
        audio_to_process = self.upload_to_gemini(path_audio2process, mime_type="audio/wav")
        res = self.chat_session.send_message(audio_to_process, 
                                         stream=True
                                        )
        for chuck in res:
            yield chuck.text
        # res =  self.llm_model.generate_content(self.few_shot_examples + ["input: " + input_text + " output: "])
        # return res.text

st.set_page_config(page_title="HebSpeech2Text")

st.markdown("""
<style>
textarea {
  unicode-bidi:bidi-override;
  direction: RTL;
}
</style>
    """, unsafe_allow_html=True)
st.markdown("""
<style>
h1 {
  unicode-bidi:bidi-override;
  direction: RTL;
}
p {
  unicode-bidi:bidi-override;
  direction: RTL;
}
</style>
    """, unsafe_allow_html=True)


def load_yaml_file_st(uploaded_file):
  """Loads a YAML file from an uploaded file object and returns its contents as a Python object.

  Args:
    uploaded_file: The uploaded file object.

  Returns:
    The parsed YAML content as a Python object.
  """

  if uploaded_file is not None:
    content = uploaded_file.read()
    try:
      data = yaml.safe_load(content)
      return data
    except yaml.YAMLError as exc:
      st.error(f"Error parsing YAML: {exc}")
      return None
  else:
    st.warning("Please upload a YAML file")
    return None


def save_audio(audio_data, filename="recorded_audio_seg_2.wav"):
    with open(filename, mode='wb') as f:
        f.write(audio_data)
    # write(filename, sample_rate, audio_data)
    return filename



def main():
    st.title('אודיו לטקסט עברית')#+ 'Heb speech2text'[::-1])
    global cfg
    cfg = ''
    cfg_ = st.sidebar.file_uploader("  פה מעלים קובץ קונפיגורציה", type=['ymal','yml'])
    if cfg_ is not None:
        cfg = load_yaml_file_st(cfg_)
    
    # if cfg:
    #     instructions = st.sidebar.text_area('הנחיות למודל השפה',cfg['instruction'] if 'instruction' in cfg.keys() else '' if cfg else '')
    
    if cfg:
        audio_transcription_model = Speech2Text_LLM(api_key=cfg['api_key'])
        col2, col1 = st.columns(2)
        
        apply_speech2text = st.button('לבצע המרה אודיו לטקסט')
        if True:#with col2:
            wav_audio_data = st_audiorec() #audiorecorder(start_prompt="", stop_prompt="", pause_prompt="")
            if wav_audio_data:
                saved_audio_file_name = 'recorded_audio_seg.wav'
                save_audio(wav_audio_data, saved_audio_file_name)
                # wav_audio_data.export(saved_audio_file_name, format='wav')
                # saved_file = save_audio(wav_audio_data)
                # col1.audio(saved_audio_file_name, format="audio/wav")
                
        
        transcription_area = st.empty()
        if wav_audio_data or apply_speech2text:
            with st.spinner('ממיר אודיו לטקסט'):
                # Transcribe audio
                transcription = audio_transcription_model.apply(saved_audio_file_name)
        
                res = ''
                for chuck in transcription:    
                    res += chuck
                    transcription_area.text_area("פלט", value=res, height=200)
                    # st.write(chuck)
                    # st.session_state.transcribed_text += chuck
                st.success('ההמרה הסתיימה!')
                
if __name__ =="__main__":
    main()
