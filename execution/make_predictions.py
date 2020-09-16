from utilities import make_path, check_existence, load_data, output_abusive_intent, save_vector
from utilities.pre_processing import runtime_clean
from config import dataset, fast_text_model
from fasttext import load_model as ft_load
from model.core import RealtimeEmbedding, AttentionWithContext, predict_abusive_intent
from keras.models import load_model as keras_load
from numpy import argsort


base = make_path('data/')
embedding_path = base / 'model' / (fast_text_model + '.bin')
model_dir = base / 'model' / 'production/'
context_path = base / 'source' / (dataset + '_clean.csv')
target_dir = base / 'predictions/'

check_existence([embedding_path, model_dir, context_path])
print('Config complete.')

embedding_model = ft_load(str(embedding_path))
model = keras_load(model_dir, custom_objects={'AttentionWithContext': AttentionWithContext})
print('Loaded models.')

contexts = load_data(context_path)['contexts'].values
contexts = runtime_clean(contexts)
realtime_data = RealtimeEmbedding(embedding_model, contexts)
print('Loaded and prepared data.')

prediction_bundle = predict_abusive_intent(realtime_data, model)

bundle_headers = ['abuse', 'intent', 'abusive_intent']
for prediction, header in zip(prediction_bundle, bundle_headers):
    target_path = target_dir / (header + '.csv')
    save_vector(prediction, target_path)

sample_size = 25
indexes = reversed(argsort(prediction_bundle[-1])[-sample_size:])
output_abusive_intent(indexes, prediction_bundle, contexts)
