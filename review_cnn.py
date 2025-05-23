from tensorflow.keras.models import load_model

# Tải mô hình đã huấn luyện
model = load_model('final_model.h5')

# Hiển thị kiến trúc
model.summary()