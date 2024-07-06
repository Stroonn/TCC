import os
os.environ['TF_USE_LEGACY_KERAS'] = 'True'
import ktrain
from ktrain import vision as vis
import re

from make_model_prediction import make_model_prediction

def build_model():
    operation = 'mean'
    model_name = input("Enter model name: ")
    bean_parameter = input("Choose the bean parameter (L, a or b): ")
    dataset = "CD" + input("Choose the dataset (1, 2 or 3)")
    epoch = input("Choose the epochs")
    lr = input("Choose the learning rate")
    
    DATADIR = "features/" + dataset + "/" + bean_parameter + '_' + operation + '/'
    PATTERN = r'(\d+(\.\d+)?)\.jpg$'
    p = re.compile(PATTERN)
    
    data_aug = vis.get_data_aug(horizontal_flip=True, vertical_flip=True)
    (train_data, val_data, preproc) = vis.images_from_fname(DATADIR, pattern = PATTERN, data_aug = data_aug, val_pct=0.33, is_regression=True, random_state=42)  

    model = vis.image_regression_model(model_name, train_data, val_data)

    learner = ktrain.get_learner(model=model, train_data=train_data, val_data=val_data, workers=8, use_multiprocessing=False, batch_size=16)
    learner.autofit(float(lr), int(epoch), reduce_on_plateau=5)

    predictor = ktrain.get_predictor(learner.model, preproc)
    predictor.save("models/finals/" + model_name + "/" + dataset + "/" + bean_parameter + "/" + str(epoch) + "_" + str(lr))
    
    make_model_prediction(predictor, bean_parameter, lr, epoch, dataset, model_name)

    print("Model built successfully.")
        
    # prediction = predictor.predict_filename(img_path='C:/Users/ogabr/OneDrive/Documentos/TCC/datasets/CD3/images/a1.jpg')
    # print(prediction)
