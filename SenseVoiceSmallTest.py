# from model import SenseVoiceSmall
# from funasr.utils.postprocess_utils import rich_transcription_postprocess

# model_dir = "/media/verachen/e0f7a88c-ad43-4736-8829-4d06e5ed8f4f/model/Voice/SenseVoice/SenseVoiceSmall"
# m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
# m.eval()

# res = m.inference(
#     data_in=f"{kwargs['model_path']}/example/en.mp3",
#     language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=False,
#     ban_emo_unk=False,
#     **kwargs,
# )

# text = rich_transcription_postprocess(res[0][0]["text"])
# print(text)

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "/media/verachen/e0f7a88c-ad43-4736-8829-4d06e5ed8f4f/model/Voice/SenseVoice/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",    
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)