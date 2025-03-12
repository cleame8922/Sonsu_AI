import h5py

model_path = 'models/model1.h5'  # 모델 경로

with h5py.File(model_path, 'r') as f:
    if 'actions' in f.attrs:
        actions = f.attrs['actions']
        print("저장된 actions 리스트:", actions)
    else:
        print("모델 파일에 actions 정보가 없습니다.")

