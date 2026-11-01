import numpy as np
import os
import time
from sklearn.decomposition import PCA
from numpy.lib.stride_tricks import sliding_window_view


def extract_spatial_patches_vectorized(I, patch_size=3):
    """
    Versión vectorizada y optimizada para la extracción de parches secuenciales.
    """
    M, N, B = I.shape
    offset = patch_size // 2

    I_padded = np.pad(I, ((offset, offset), (offset, offset), (0, 0)), mode='constant')
    patches = sliding_window_view(I_padded, window_shape=(patch_size, patch_size, B))
    J_patches = patches.reshape(M * N, patch_size * patch_size, B)
    return J_patches


class DimensionalityReducer:
    def fit_transform(self, data, n_components):
        raise NotImplementedError

    @staticmethod
    def create(method_name, n_components, **kwargs):
        method = method_name.upper()
        if method == 'PCA':
            return PCAReducer(n_components)
        elif method in ('AE', 'AUTOENCODER'):
            return AEReducer(n_components, **kwargs)
        elif method == 'AE_PATCH':
            return AEPatchReducer(n_components, **kwargs)
        elif method == 'VIT':
            return ViTReducer(n_components, **kwargs)
        elif method == 'MAE':
            return MAEReducer(n_components, **kwargs)
        elif method == 'DINO':
            return DINOReducer(n_components, **kwargs)
        else:
            raise ValueError(
                f"Método '{method_name}' no soportado. Opciones: PCA, AE, VIT, MAE, DINO"
            )


class PCAReducer(DimensionalityReducer):
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = None

    def fit_transform(self, data, n_components=None, train_mask=None):
        nc = n_components or self.n_components
        data_2d = data.reshape(-1, data.shape[-1]) if data.ndim == 3 else data
        
        self.pca = PCA(n_components=nc, svd_solver='auto', whiten=True, random_state=42)
        if train_mask is not None:
            train_data = data_2d[train_mask.ravel()]
            self.pca.fit(train_data)
            return self.pca.transform(data_2d)
        return self.pca.fit_transform(data_2d)

    def transform(self, data):
        if self.pca is None:
            raise ValueError("PCA model has not been fitted yet.")
        data_2d = data.reshape(-1, data.shape[-1]) if data.ndim == 3 else data
        return self.pca.transform(data_2d)




def create_array_sequence(tf, x, y, batch_size=4096):
    class ArraySequence(tf.keras.utils.Sequence):
        def __init__(self, x_data, y_data, b_size):
            super().__init__()
            self.x = x_data
            self.y = y_data
            self.b_size = b_size
        def __len__(self):
            return int(np.ceil(len(self.x) / self.b_size))
        def __getitem__(self, idx):
            bx = self.x[idx * self.b_size:(idx + 1) * self.b_size]
            by = self.y[idx * self.b_size:(idx + 1) * self.b_size]
            return bx, by
    return ArraySequence(x, y, batch_size)


# --- Base class for Deep Learning Reducers ---
class DeepReducer(DimensionalityReducer):
    def __init__(self, n_components, **kwargs):
        self.n_components = n_components
        self.device = kwargs.get('device', 'CPU')
        self.epochs = kwargs.get('epochs', 100) # Se puede subir por config si se desea mayor precisión
        self.batch_size = kwargs.get('batch_size', 128)
        self.patch_size = kwargs.get('patch_size', 3)
        self.patience = kwargs.get('patience', 10)
        self.tf = None

    def _setup_tf(self):
        if self.tf is None:
            # Desactivar logs del backend de C++ de TensorFlow y optimizaciones oneDNN
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            
            if self.device.upper() == 'CPU':
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            import tensorflow as tf
            
            if self.device.upper() != 'CPU':
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        print(e)
            
            tf.keras.utils.set_random_seed(42)
            self.tf = tf
            self._register_custom_layers()
        return self.tf

    def _register_custom_layers(self):
        tf = self.tf
        
        class PositionalEmbedding(tf.keras.layers.Layer):
            def __init__(self, num_tokens, embed_dim, **kwargs):
                super().__init__(**kwargs)
                self.num_tokens = num_tokens
                self.embed_dim = embed_dim

            def build(self, input_shape):
                self.pos_emb = self.add_weight(
                    name="pos_emb",
                    shape=(1, self.num_tokens + 1, self.embed_dim),
                    initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                    trainable=True
                )

            def call(self, inputs):
                return inputs + self.pos_emb

        class CLSToken(tf.keras.layers.Layer):
            def __init__(self, embed_dim, **kwargs):
                super().__init__(**kwargs)
                self.embed_dim = embed_dim

            def build(self, input_shape):
                self.cls_token = self.add_weight(
                    name="cls_token",
                    shape=(1, 1, self.embed_dim),
                    initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                    trainable=True
                )

            def call(self, inputs):
                batch_size = tf.shape(inputs)[0]
                broadcast_cls = tf.broadcast_to(self.cls_token, [batch_size, 1, self.embed_dim])
                return tf.concat([broadcast_cls, inputs], axis=1)

        class PatchMasking(tf.keras.layers.Layer):
            def __init__(self, mask_ratio=0.75, **kwargs):
                super().__init__(**kwargs)
                self.mask_ratio = mask_ratio

            def call(self, inputs, training=None):
                batch_size = tf.shape(inputs)[0]
                num_tokens = tf.shape(inputs)[1]
                
                if not training:
                    # En inferencia pasamos todos los tokens y devolvemos un dummy tensor de indices
                    dummy_indices = tf.zeros((batch_size, num_tokens), dtype=tf.int32)
                    return inputs, dummy_indices
                
                num_masked = tf.cast(tf.math.round(tf.cast(num_tokens, tf.float32) * self.mask_ratio), tf.int32)
                
                rand_indices = tf.argsort(tf.random.uniform(shape=(batch_size, num_tokens)), axis=-1)
                visible_indices = rand_indices[:, num_masked:]
                visible_tokens = tf.gather(inputs, visible_indices, batch_dims=1)
                return visible_tokens, visible_indices
        
        self.PositionalEmbedding = PositionalEmbedding
        self.CLSToken = CLSToken
        self.PatchMasking = PatchMasking

    def fit_transform(self, data, n_components=None):
        raise NotImplementedError


class AEReducer(DeepReducer):
    def fit_transform(self, data, n_components=None, train_mask=None):
        tf = self._setup_tf()
        nc = n_components or self.n_components
        
        data_2d = data.reshape(-1, data.shape[-1]) if data.ndim == 3 else data
        input_dim = data_2d.shape[-1]
        
        inputs = tf.keras.Input(shape=(input_dim,), name="Inputs")
        encoder = tf.keras.layers.Dense(units=nc, activation='relu',
                                        kernel_initializer=tf.keras.initializers.HeNormal(),
                                        kernel_regularizer=tf.keras.regularizers.OrthogonalRegularizer(factor=0.01, mode="rows"),
                                        name="Encoder")(inputs)
        decoder = tf.keras.layers.Dense(units=input_dim, activation='sigmoid',
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                        name="Decoder")(encoder)
        
        model = tf.keras.Model(inputs=inputs, outputs=decoder)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        
        early_callback= tf.keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=self.patience, restore_best_weights=True)
        
        train_data = data_2d[train_mask.ravel()] if train_mask is not None else data_2d
        batch_sz = max(self.batch_size, 4096) if train_data.shape[0] > 100000 else self.batch_size
        
        train_gen = create_array_sequence(tf, train_data, train_data, batch_size=batch_sz)
        
        print(f"       [AE] Entrenando Autoencoder ({self.epochs} epochs en {self.device}, batch_size={batch_sz})...")
        model.fit(train_gen, epochs=self.epochs, verbose=0, callbacks=[early_callback])
        
        encoder_model = tf.keras.Model(inputs=inputs, outputs=encoder)
        self.model = encoder_model
        return encoder_model.predict(data_2d, batch_size=batch_sz, verbose=0)

    def transform(self, data):
        if self.model is None:
            raise ValueError("AE model has not been fitted yet.")
        data_2d = data.reshape(-1, data.shape[-1]) if data.ndim == 3 else data
        return self.model.predict(data_2d, batch_size=self.batch_size, verbose=0)


class AEPatchReducer(DeepReducer):
    def fit_transform(self, data, n_components=None, train_mask=None):
        tf = self._setup_tf()
        nc = n_components or self.n_components
        
        if data.ndim != 3:
            raise ValueError("AE_PATCH requiere el cubo de datos 3D (rows, cols, bands).")
            
        data_for_train = data.copy()
        if train_mask is not None:
            data_for_train[~train_mask] = 0.0 # Prevents Data Leakage
            
        Np = data.shape[0] * data.shape[1]
        train_patches = extract_spatial_patches_vectorized(data_for_train, self.patch_size).reshape((Np, -1))
        train_data = train_patches[train_mask.ravel()] if train_mask is not None else train_patches
        input_dim = train_patches.shape[1]
        
        inputs = tf.keras.Input(shape=(input_dim,), name="Inputs")
        encoder = tf.keras.layers.Dense(units=nc, activation='relu',
                                        kernel_initializer=tf.keras.initializers.HeNormal(),
                                        kernel_regularizer=tf.keras.regularizers.OrthogonalRegularizer(factor=0.01, mode="rows"),
                                        name="Encoder")(inputs)
        decoder = tf.keras.layers.Dense(units=input_dim, activation='sigmoid',
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                        name="Decoder")(encoder)
        
        model = tf.keras.Model(inputs=inputs, outputs=decoder)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        
        early_callback= tf.keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=self.patience, restore_best_weights=True)
        
        batch_sz = max(self.batch_size, 4096) if train_data.shape[0] > 100000 else self.batch_size
        train_gen = create_array_sequence(tf, train_data, train_data, batch_size=batch_sz)
        print(f"       [AE_PATCH] Entrenando Autoencoder de Parches ({self.epochs} epochs en {self.device}, batch_size={batch_sz})...")
        model.fit(train_gen, epochs=self.epochs, verbose=0, callbacks=[early_callback])
        
        encoder_model = tf.keras.Model(inputs=inputs, outputs=encoder)
        self.model = encoder_model
        
        full_patches = extract_spatial_patches_vectorized(data, self.patch_size).reshape((Np, input_dim))
        return encoder_model.predict(full_patches, batch_size=self.batch_size, verbose=0)

    def transform(self, data):
        if self.model is None:
            raise ValueError("AE_PATCH model has not been fitted yet.")
        if data.ndim == 3:
            Np = data.shape[0] * data.shape[1]
            full_patches = extract_spatial_patches_vectorized(data, self.patch_size).reshape((Np, -1))
            return self.model.predict(full_patches, batch_size=self.batch_size, verbose=0)
        else:
            n_signatures = data.shape[0]
            num_tokens = self.patch_size * self.patch_size
            tiled = np.repeat(data[:, np.newaxis, :], num_tokens, axis=1)
            tiled = tiled.reshape(n_signatures, -1)
            return self.model.predict(tiled, batch_size=self.batch_size, verbose=0)


class ViTReducer(DeepReducer):
    def fit_transform(self, data, n_components=None, train_mask=None):
        tf = self._setup_tf()
        nc = n_components or self.n_components
        
        if data.ndim != 3:
            raise ValueError("ViT requiere el cubo de datos 3D (rows, cols, bands).")
            
        data_for_train = data.copy()
        if train_mask is not None:
            data_for_train[~train_mask] = 0.0 # Prevents Data Leakage
        
        train_patches = extract_spatial_patches_vectorized(data_for_train, self.patch_size)
        train_data = train_patches[train_mask.ravel()] if train_mask is not None else train_patches
        
        num_tokens = self.patch_size * self.patch_size
        input_dim = data.shape[-1]
        embed_dim = 48
        
        inputs = tf.keras.Input(shape=(num_tokens, input_dim), name="Patch_Input")
        x = tf.keras.layers.Dense(units=embed_dim, name="Linear_Projection")(inputs)
        x = self.CLSToken(embed_dim=embed_dim, name="Add_CLS_Token")(x)
        x = self.PositionalEmbedding(num_tokens=num_tokens, embed_dim=embed_dim, name="Pos_Emb")(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=embed_dim, dropout=0.1)(norm_1, norm_1)
        x = tf.keras.layers.Add()([x, attn_output])

        norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = tf.keras.layers.Dense(units=96, activation='gelu')(norm_2)
        ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
        ffn_output = tf.keras.layers.Dense(units=embed_dim)(ffn_output)
        x = tf.keras.layers.Add()([x, ffn_output])

        cls_output = tf.keras.layers.Lambda(lambda t: t[:, 0], name="Extract_CLS")(x)
        norm_cls = tf.keras.layers.LayerNormalization(epsilon=1e-6)(cls_output)
        latent_z = tf.keras.layers.Dense(units=nc, name="Latent_Space_Z")(norm_cls)

        dec = tf.keras.layers.Dense(units=embed_dim, activation='gelu')(latent_z)
        dec = tf.keras.layers.RepeatVector(num_tokens)(dec)
        outputs = tf.keras.layers.Dense(units=input_dim, activation='sigmoid')(dec)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
        
        batch_sz = max(self.batch_size, 4096) if train_data.shape[0] > 100000 else self.batch_size
        train_gen = create_array_sequence(tf, train_data, train_data, batch_size=batch_sz)
        print(f"       [ViT] Entrenando ViT ({self.epochs} epochs en {self.device}, batch_size={batch_sz})...")
        model.fit(train_gen, epochs=self.epochs, verbose=0, callbacks=[early_callback])
        
        encoder_model = tf.keras.Model(inputs=inputs, outputs=latent_z)
        self.model = encoder_model
        
        full_patches = extract_spatial_patches_vectorized(data, self.patch_size)
        return encoder_model.predict(full_patches, batch_size=self.batch_size, verbose=0)

    def transform(self, data):
        if self.model is None:
            raise ValueError("ViT model has not been fitted yet.")
        if data.ndim == 3:
            full_patches = extract_spatial_patches_vectorized(data, self.patch_size)
            return self.model.predict(full_patches, batch_size=self.batch_size, verbose=0)
        else:
            n_signatures = data.shape[0]
            num_tokens = self.patch_size * self.patch_size
            tiled = np.repeat(data[:, np.newaxis, :], num_tokens, axis=1)
            return self.model.predict(tiled, batch_size=self.batch_size, verbose=0)


class MAEReducer(DeepReducer):
    def fit_transform(self, data, n_components=None, train_mask=None):
        tf = self._setup_tf()
        nc = n_components or self.n_components
        
        if data.ndim != 3:
            raise ValueError("MAE requiere el cubo de datos 3D (rows, cols, bands).")
            
        data_for_train = data.copy()
        if train_mask is not None:
            data_for_train[~train_mask] = 0.0 # Prevents Data Leakage
            
        train_patches = extract_spatial_patches_vectorized(data_for_train, self.patch_size)
        train_data = train_patches[train_mask.ravel()] if train_mask is not None else train_patches
        
        num_tokens = self.patch_size * self.patch_size
        input_dim = data.shape[-1]
        embed_dim = 64
        
        inputs = tf.keras.Input(shape=(num_tokens, input_dim), name="Patch_Input")
        x = tf.keras.layers.Dense(embed_dim)(inputs) 
        visible_tokens, _ = self.PatchMasking(mask_ratio=0.5)(x)
        
        norm_1 = tf.keras.layers.LayerNormalization()(visible_tokens)
        attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim)(norm_1, norm_1)
        x = tf.keras.layers.Add()([visible_tokens, attn])
        
        latent_z = tf.keras.layers.GlobalAveragePooling1D(name="Latent_Space_Z")(x)
        latent_z = tf.keras.layers.Dense(nc, name="Latent_Dense")(latent_z)

        dec = tf.keras.layers.RepeatVector(num_tokens)(latent_z)
        dec = tf.keras.layers.Dense(embed_dim, activation='gelu')(dec)
        outputs = tf.keras.layers.Dense(input_dim, activation='sigmoid', name="Reconstruction")(dec)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
        
        batch_sz = max(self.batch_size, 4096) if train_data.shape[0] > 100000 else self.batch_size
        train_gen = create_array_sequence(tf, train_data, train_data, batch_size=batch_sz)
        print(f"       [MAE] Entrenando MAE ({self.epochs} epochs en {self.device}, batch_size={batch_sz})...")
        model.fit(train_gen, epochs=self.epochs, verbose=0, callbacks=[early_callback])
        
        encoder_model = tf.keras.Model(inputs=inputs, outputs=latent_z)
        self.model = encoder_model
        
        full_patches = extract_spatial_patches_vectorized(data, self.patch_size)
        return encoder_model.predict(full_patches, batch_size=self.batch_size, verbose=0)

    def transform(self, data):
        if self.model is None:
            raise ValueError("MAE model has not been fitted yet.")
        if data.ndim == 3:
            full_patches = extract_spatial_patches_vectorized(data, self.patch_size)
            return self.model.predict(full_patches, batch_size=self.batch_size, verbose=0)
        else:
            n_signatures = data.shape[0]
            num_tokens = self.patch_size * self.patch_size
            tiled = np.repeat(data[:, np.newaxis, :], num_tokens, axis=1)
            return self.model.predict(tiled, batch_size=self.batch_size, verbose=0)


class DINOReducer(DeepReducer):
    def fit_transform(self, data, n_components=None, train_mask=None):
        tf = self._setup_tf()
        nc = n_components or self.n_components
        
        if data.ndim != 3:
            raise ValueError("DINO requiere el cubo de datos 3D (rows, cols, bands).")
            
        data_for_train = data.copy()
        if train_mask is not None:
            data_for_train[~train_mask] = 0.0 # Prevents Data Leakage
            
        train_patches = extract_spatial_patches_vectorized(data_for_train, self.patch_size)
        train_data = train_patches[train_mask.ravel()] if train_mask is not None else train_patches
        
        # Split train_data into train and validation sets (90% train, 10% validation)
        num_train_samples = len(train_data)
        indices = np.arange(num_train_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(0.9 * num_train_samples)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        if len(val_idx) == 0 and num_train_samples > 1:
            train_idx = indices[:-1]
            val_idx = indices[-1:]
        elif num_train_samples <= 1:
            train_idx = indices
            val_idx = indices

        X_T = train_data[train_idx]
        X_v = train_data[val_idx]
        
        num_tokens = self.patch_size * self.patch_size
        input_dim = data.shape[-1]
        embed_dim = 128
        
        @tf.function
        def augment(images, apply_occlusion=False):
            x = tf.reshape(images, [-1, self.patch_size, self.patch_size, input_dim])
            x = tf.image.random_flip_left_right(x)
            x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            
            scale = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.99, 1.01)
            x = (x * scale) + tf.random.normal(tf.shape(x), stddev=0.01)
            x = tf.reshape(x, [-1, num_tokens, input_dim])
            
            if apply_occlusion:
                mask = np.ones((1, num_tokens, 1), dtype=np.float32)
                mask[:, num_tokens//2, :] = 0.0 # Occlude center
                x = x * mask
            return x

        def build_vit():
            inputs = tf.keras.Input(shape=(num_tokens, input_dim))
            x = tf.keras.layers.Dense(embed_dim)(inputs)
            x = self.CLSToken(embed_dim)(x) 
            x = self.PositionalEmbedding(num_tokens, embed_dim)(x) 
            
            norm = tf.keras.layers.LayerNormalization()(x)
            attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim)(norm, norm)
            x = tf.keras.layers.Add()([x, attn])
            
            cls_extract = tf.keras.layers.Lambda(lambda t: t[:, 0], name="CLS_Extract")(x)
            latent = tf.keras.layers.Dense(nc, name="Latent_Space")(cls_extract)
            return tf.keras.Model(inputs, latent)

        student = build_vit()
        teacher = build_vit()
        teacher.set_weights(student.get_weights())

        K_DIM = 4096
        def build_head():
            return tf.keras.Sequential([
                tf.keras.layers.Input(shape=(nc,)),
                tf.keras.layers.Dense(512, activation='gelu'),
                tf.keras.layers.Dense(K_DIM)
            ])
        
        s_head, t_head = build_head(), build_head()
        t_head.set_weights(s_head.get_weights())

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        teacher_center = tf.Variable(tf.zeros((1, K_DIM)), trainable=False)
        
        @tf.function(jit_compile=True)
        def train_step(batch):
            v_teacher = augment(batch, apply_occlusion=False)
            v_student = augment(batch, apply_occlusion=True)
            
            with tf.GradientTape() as tape:
                t_logits = t_head(teacher(v_teacher, training=False))
                s_logits = s_head(student(v_student, training=True))
                t_probs = tf.nn.softmax((t_logits - teacher_center) / 0.01)
                s_probs = tf.nn.log_softmax(s_logits / 0.1)
                loss = tf.reduce_mean(-tf.reduce_sum(t_probs * s_probs, axis=-1))

            grads = tape.gradient(loss, student.trainable_variables + s_head.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables + s_head.trainable_variables))

            for s_v, t_v in zip(student.variables + s_head.variables, teacher.variables + t_head.variables):
                t_v.assign(t_v * 0.999 + s_v * 0.001)
                
            teacher_center.assign(teacher_center * 0.9 + tf.reduce_mean(t_logits, axis=0, keepdims=True) * 0.1)
            return loss

        @tf.function(jit_compile=True)
        def val_step(v_batch):
            v_t_logits = t_head(teacher(v_batch, training=False))
            v_s_logits = s_head(student(v_batch, training=False))
            v_loss = tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(v_t_logits/0.02) * tf.nn.log_softmax(v_s_logits/0.1), axis=-1))
            return v_loss

        with tf.device('/CPU:0'):
            train_ds = tf.data.Dataset.from_tensor_slices(X_T.astype('float32')).shuffle(buffer_size=min(10000, len(X_T)), seed=42).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices(X_v.astype('float32')).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        best_val_loss = float('inf')
        patience = self.patience
        wait = 0
        best_weights = teacher.get_weights()

        print(f"       [DINO] Entrenando DINO ({self.epochs} epochs en {self.device})...")
        
        for epoch in range(self.epochs):
            train_losses = []
            for batch in train_ds:
                loss = train_step(batch)
                train_losses.append(loss)
            
            val_losses = []
            for v_batch in val_ds:
                v_loss = val_step(v_batch)
                val_losses.append(v_loss)

            cur_train_loss = np.mean(train_losses)
            cur_val_loss = np.mean(val_losses)

            report_interval = max(1, self.epochs // 5)
            if (epoch + 1) % report_interval == 0 or (epoch + 1) == self.epochs:
                print(f"       [DINO] Epoch {epoch+1:03d}/{self.epochs} - loss: {cur_train_loss:.6f} - val_loss: {cur_val_loss:.6f}")

            if cur_val_loss < best_val_loss:
                best_val_loss = cur_val_loss
                wait = 0
                best_weights = teacher.get_weights()
            else:
                wait += 1
                if wait >= patience:
                    print(f"       [DINO] Early stopping en epoch {epoch+1}")
                    break

        teacher.set_weights(best_weights)
        self.model = teacher
        
        full_patches = extract_spatial_patches_vectorized(data, self.patch_size)
        return teacher.predict(full_patches, batch_size=self.batch_size, verbose=0)

    def transform(self, data):
        if self.model is None:
            raise ValueError("DINO model has not been fitted yet.")
        if data.ndim == 3:
            full_patches = extract_spatial_patches_vectorized(data, self.patch_size)
            return self.model.predict(full_patches, batch_size=self.batch_size, verbose=0)
        else:
            n_signatures = data.shape[0]
            num_tokens = self.patch_size * self.patch_size
            tiled = np.repeat(data[:, np.newaxis, :], num_tokens, axis=1)
            return self.model.predict(tiled, batch_size=self.batch_size, verbose=0)
