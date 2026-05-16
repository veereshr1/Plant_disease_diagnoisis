import traceback
try:
    import tensorflow as tf
    print('ok', tf.__version__)
    import tensorflow.python as tpp
    print('ok2', tpp)
except Exception:
    traceback.print_exc()
