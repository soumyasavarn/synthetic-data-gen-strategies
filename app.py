import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from torchvision.utils import save_image
import scipy.linalg
from sklearn.preprocessing import MinMaxScaler
import io

# --- Model Definition ---
def conv_block(in_c, out_c, norm=True):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
    if norm: layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def deconv_block(in_c, out_c, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    ]
    if dropout: layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

def denoise_image_pil(image_np):
    image_pil = Image.fromarray(image_np)
    denoised_pil = image_pil.filter(ImageFilter.MedianFilter(size=3))
    return np.array(denoised_pil)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = conv_block(3, 64, norm=False)
        self.d2 = conv_block(64, 128)
        self.d3 = conv_block(128, 256)
        self.d4 = conv_block(256, 512)
        self.d5 = conv_block(512, 512)
        self.d6 = conv_block(512, 512)
        self.d7 = conv_block(512, 512)
        self.d8 = conv_block(512, 512)
        self.u1 = deconv_block(512, 512, dropout=True)
        self.u2 = deconv_block(1024, 512, dropout=True)
        self.u3 = deconv_block(1024, 512, dropout=True)
        self.u4 = deconv_block(1024, 512)
        self.u5 = deconv_block(1024, 256)
        self.u6 = deconv_block(512, 128)
        self.u7 = deconv_block(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1)
        d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5)
        d7 = self.d7(d6); d8 = self.d8(d7)
        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7],1))
        u3 = self.u3(torch.cat([u2, d6],1))
        u4 = self.u4(torch.cat([u3, d5],1))
        u5 = self.u5(torch.cat([u4, d4],1))
        u6 = self.u6(torch.cat([u5, d3],1))
        u7 = self.u7(torch.cat([u6, d2],1))
        return self.final(torch.cat([u7, d1],1))

# --- Cartoon Generation Function ---
def generate_cartoon_image(input_pil_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    generator.load_state_dict(torch.load("generator_120.pth", map_location=device))
    generator.eval()

    # Preprocess input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust if your training used a different size
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # [-1,1] normalization
    ])
    img_tensor = transform(input_pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        cartoon_tensor = generator(img_tensor).cpu()
    cartoon_tensor = (cartoon_tensor + 1) / 2.0  # [0,1] range
    cartoon_np = cartoon_tensor.squeeze(0).permute(1, 2, 0).numpy()
    cartoon_np = (cartoon_np * 255).clip(0,255).astype(np.uint8)
    return cartoon_np

# --- Custom KNN Image Imputation ---
def knn_image_imputation(image, missing_rate, n_neighbors=100):
    """
    Impute missing pixels using KNN based on 2D spatial relationships.
    
    Args:
        image: Input image as numpy array (h, w, c)
        missing_rate: Percentage of pixels to mask (0.0-1.0)
        n_neighbors: Number of nearest neighbors to use for imputation
        
    Returns:
        Imputed image as numpy array
    """
    h, w, c = image.shape
    imputed_image = image.copy()
    
    np.random.seed(42)  # For reproducibility
    mask = np.random.rand(h, w) < missing_rate
    
    for channel_idx in range(c):
        channel = image[:, :, channel_idx].copy()
        
        # Create coordinate grid for all pixels
        y_coords, x_coords = np.meshgrid(range(h), range(w), indexing='ij')
        
        # Flatten arrays for KNN processing
        all_coords = np.column_stack([y_coords.ravel(), x_coords.ravel()])
        all_values = channel.ravel()
        
        # Identify missing and known pixels
        missing_indices = np.where(mask.ravel())[0]
        known_indices = np.where(~mask.ravel())[0]
        
        if len(missing_indices) == 0 or len(known_indices) == 0:
            continue
            
        missing_coords = all_coords[missing_indices]
        known_coords = all_coords[known_indices]
        known_values = all_values[known_indices]
        
        actual_n_neighbors = min(n_neighbors, len(known_coords))
        if actual_n_neighbors == 0:
            continue
            
        nbrs = NearestNeighbors(n_neighbors=actual_n_neighbors, algorithm='auto', metric='euclidean')
        nbrs.fit(known_coords)
        
        distances, indices = nbrs.kneighbors(missing_coords)
        
        neighbor_values = known_values[indices]
        imputed_values = stats.mode(neighbor_values, axis=1)[0].flatten()
        
        channel_flat = channel.ravel()
        channel_flat[missing_indices] = imputed_values
        imputed_image[:, :, channel_idx] = channel_flat.reshape(h, w)
    
    return imputed_image

# --- Improved Tabular GAN for CSV data ---
class CTGAN(nn.Module):
    """Conditional Tabular GAN - better for tabular data than vanilla GAN"""
    def __init__(self, input_dim, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Generator with residual connections
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        # Discriminator with spectral normalization for stability
        self.discriminator = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, 1)),
            nn.Sigmoid()
        )
    
    def generate(self, batch_size):
        device = next(self.generator.parameters()).device
        z = torch.randn(batch_size, self.latent_dim).to(device)
        return self.generator(z)

# --- Streamlit UI ---
st.set_page_config(
    page_title="File Upload for GAN Processing",
    layout="wide"
)

with st.sidebar:
    st.info("Built by Soumya Savarn")
    st.markdown("### Settings")
    denoising_method = st.radio(
        "Denoising Method",
        ["Median Filter"],
        index=0
    )
    
    if denoising_method == "Median Filter":
        filter_size = st.slider("Median Filter Size", 1, 9, 3, 2)
    
    st.markdown("---")

st.title("Synthetic Data generator using GANs")
st.subheader("Upload CSVs or images")

pages = st.tabs(["Upload Files", "About"])

with pages[0]:
    st.header("Upload Files")
    
    # Place upload widget and test data selection side by side
    col_upload, col_test = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader("Upload your CSV or Image", type=["csv", "jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['file_type'] = uploaded_file.name.split('.')[-1].lower()

    with col_test:
        st.markdown("### Or select from test data")
        test_data_option = st.selectbox(
            "Choose test data type",
            ["CSV Examples", "Image Examples"],
            help="Select from available test datasets"
        )
        
        if test_data_option == "CSV Examples":
            test_csv_files = ["iris.csv", "student_depression_dataset.csv"]
            selected_test_file = st.selectbox("Select test CSV:", test_csv_files, key="test_csv")
            if st.button("Load Test CSV"):
                try:
                    test_file_path = f"testing_data/csv/{selected_test_file}"
                    with open(test_file_path, 'rb') as file:
                        file_content = file.read()
                    st.session_state['uploaded_file'] = io.BytesIO(file_content)
                    st.session_state['file_type'] = 'csv'
                    st.success(f"Loaded test CSV: {selected_test_file}")
                except Exception as e:
                    st.error(f"Error loading test file: {e}")
                    
        elif test_data_option == "Image Examples":
            test_image_files = [f"15{i}.jpg" for i in range(90, 100)]
            selected_test_file = st.selectbox("Select test image:", test_image_files, key="test_img")
            if st.button("Load Test Image"):
                try:
                    test_file_path = f"testing_data/images/{selected_test_file}"
                    with open(test_file_path, 'rb') as file:
                        file_content = file.read()
                    st.session_state['uploaded_file'] = io.BytesIO(file_content)
                    st.session_state['file_type'] = 'jpg'
                    st.success(f"Loaded test image: {selected_test_file}")
                except Exception as e:
                    st.error(f"Error loading test file: {e}")

    if 'uploaded_file' in st.session_state and 'file_type' in st.session_state:
        uploaded_file = st.session_state['uploaded_file']
        file_type = st.session_state['file_type']
        
        # Reseting file pointer before processing
        uploaded_file.seek(0)

        if file_type == 'csv':
            try:
                df = pd.read_csv(uploaded_file)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                # Preview & download
                with st.expander("Data preview & extraction", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Rows", df.shape[0])
                    col2.metric("Numeric cols", len(numeric_cols))
                    csv_data = df[numeric_cols].to_csv(index=False)
                    col3.download_button(
                        "Download CSV", csv_data,
                        file_name="extracted_numerical_data.csv",
                        mime="text/csv"
                    )
                    st.dataframe(df.head())

                if numeric_cols:
                    st.session_state['gan_input_data'] = df[numeric_cols].values
                    with st.expander("GAN: train & generate synthetic data", expanded=True):
                        gan_model = st.file_uploader("Upload GAN Model (.pth) for training (optional)", type=["pth"], key="gan_model_csv")
                        
                        # Add options for different GAN models
                        gan_type = st.radio(
                            "GAN Model Type",
                            ["CTGAN (Recommended for tabular data)", "Vanilla GAN"],
                            index=0
                        )
                        
                        num_epochs = st.slider("Training Epochs", 10, 200, 50)
                        num_samples = st.slider("Number of synthetic samples to generate", 50, 500, 100)
                        
                        if st.button("Generate synthetic data"):
                            st.info(f"Training {gan_type} model on numerical data...")
                            try:
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                original = st.session_state['gan_input_data']
                                
                                # Normalize data for better GAN training
                                
                                scaler = MinMaxScaler(feature_range=(-1, 1))
                                scaled_data = scaler.fit_transform(original)
                                
                                input_dim = original.shape[1]
                                latent_dim = min(64, input_dim * 2)  # Adaptive latent dimension
                                
                                # If a GAN model is uploaded, load it
                                if gan_model is not None:
                                    with open("uploaded_gan_model.pth", "wb") as f:
                                        f.write(gan_model.getbuffer())
                                    st.write("GAN model loaded, but on-the-fly training will be performed.")

                                # Use different GAN implementations based on selection
                                if gan_type == "CTGAN (Recommended for tabular data)":
                                    model = CTGAN(input_dim=input_dim, latent_dim=latent_dim).to(device)
                                    criterion = nn.BCELoss()
                                    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
                                    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
                                    
                                    # Create progress bar
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    # Training loop with improved stability
                                    X = torch.tensor(scaled_data, dtype=torch.float32).to(device)
                                    batch_size = min(128, len(X))
                                    data_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)
                                    
                                    # Training history for plotting
                                    history = {'G_loss': [], 'D_loss': []}
                                    
                                    for epoch in range(num_epochs):
                                        g_losses, d_losses = [], []
                                        
                                        for real_data in data_loader:
                                            current_batch = real_data.size(0)
                                            valid = torch.ones(current_batch, 1).to(device)
                                            fake = torch.zeros(current_batch, 1).to(device)
                                            
                                            # Add noise to labels for label smoothing (improved stability)
                                            valid = valid - 0.1 * torch.rand_like(valid)
                                            fake = fake + 0.1 * torch.rand_like(fake)

                                            # Train Discriminator
                                            optimizer_D.zero_grad()
                                            pred_real = model.discriminator(real_data)
                                            loss_real = criterion(pred_real, valid)
                                            
                                            # Generate fake data
                                            z = torch.randn(current_batch, latent_dim).to(device)
                                            fake_data = model.generator(z)
                                            pred_fake = model.discriminator(fake_data.detach())
                                            loss_fake = criterion(pred_fake, fake)
                                            
                                            # Total discriminator loss
                                            loss_D = (loss_real + loss_fake) / 2
                                            loss_D.backward()
                                            optimizer_D.step()
                                            
                                            # Train Generator
                                            optimizer_G.zero_grad()
                                            z = torch.randn(current_batch, latent_dim).to(device)
                                            fake_data = model.generator(z)
                                            pred_fake = model.discriminator(fake_data)
                                            loss_G = criterion(pred_fake, valid)
                                            loss_G.backward()
                                            optimizer_G.step()
                                            
                                            g_losses.append(loss_G.item())
                                            d_losses.append(loss_D.item())
                                        
                                        # Update progress
                                        avg_g_loss = sum(g_losses) / len(g_losses)
                                        avg_d_loss = sum(d_losses) / len(d_losses)
                                        history['G_loss'].append(avg_g_loss)
                                        history['D_loss'].append(avg_d_loss)
                                        
                                        progress_bar.progress((epoch + 1) / num_epochs)
                                        status_text.text(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {avg_d_loss:.4f}, Loss G: {avg_g_loss:.4f}")
                                    
                                    st.success("CTGAN training complete!")
                                    
                                    # Plot training history
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    ax.plot(history['G_loss'], label='Generator Loss')
                                    ax.plot(history['D_loss'], label='Discriminator Loss')
                                    ax.set_xlabel('Epoch')
                                    ax.set_ylabel('Loss')
                                    ax.set_title('Training Losses')
                                    ax.legend()
                                    st.pyplot(fig)
                                    
                                    # After training loop completion, add detailed loss visualization
                                    st.subheader("Training Progress Visualization")
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        # Plot training losses
                                        fig, ax = plt.subplots(figsize=(8, 4))
                                        ax.plot(history['G_loss'], label='Generator', color='blue', alpha=0.7)
                                        ax.plot(history['D_loss'], label='Discriminator', color='red', alpha=0.7)
                                        ax.set_xlabel('Epoch')
                                        ax.set_ylabel('Loss')
                                        ax.set_title('Training Losses Over Time')
                                        ax.legend()
                                        ax.grid(True, alpha=0.3)
                                        st.pyplot(fig)

                                    with col2:
                                        # Plot loss ratio (G/D) to show training stability
                                        fig, ax = plt.subplots(figsize=(8, 4))
                                        loss_ratio = np.array(history['G_loss']) / (np.array(history['D_loss']) + 1e-8)
                                        ax.plot(loss_ratio, color='purple', alpha=0.7)
                                        ax.set_xlabel('Epoch')
                                        ax.set_ylabel('G/D Loss Ratio')
                                        ax.set_title('Generator/Discriminator Loss Ratio')
                                        ax.grid(True, alpha=0.3)
                                        st.pyplot(fig)

                                    # Add moving average plot for smoother visualization
                                    window_size = max(5, num_epochs // 20)  # Adaptive window size
                                    st.subheader("Smoothed Training Progress")
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    pd.Series(history['G_loss']).rolling(window=window_size).mean().plot(label='Generator (MA)', color='blue', alpha=0.7)
                                    pd.Series(history['D_loss']).rolling(window=window_size).mean().plot(label='Discriminator (MA)', color='red', alpha=0.7)
                                    ax.set_xlabel('Epoch')
                                    ax.set_ylabel('Loss (Moving Average)')
                                    ax.set_title(f'Smoothed Training Losses ({window_size}-epoch window)')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)
                                    
                                    # Generate synthetic samples
                                    model.eval()
                                    with torch.no_grad():
                                        z = torch.randn(num_samples, latent_dim).to(device)
                                        synthetic_tensor = model.generator(z)
                                        synthetic = synthetic_tensor.cpu().numpy()
                                        
                                    # Inverse transform to original scale
                                    synthetic = scaler.inverse_transform(synthetic)
                                    
                                else:  # Vanilla GAN
                                    # Define simple Generator and Discriminator models for numerical data
                                    class NumGenerator(nn.Module):
                                        def __init__(self, latent_dim, output_dim):
                                            super().__init__()
                                            self.model = nn.Sequential(
                                                nn.Linear(latent_dim, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, output_dim)
                                            )
                                        def forward(self, x):
                                            return self.model(x)

                                    class NumDiscriminator(nn.Module):
                                        def __init__(self, input_dim):
                                            super().__init__()
                                            self.model = nn.Sequential(
                                                nn.Linear(input_dim, 256),
                                                nn.LeakyReLU(0.2),
                                                nn.Linear(256, 128),
                                                nn.LeakyReLU(0.2),
                                                nn.Linear(128, 1),
                                                nn.Sigmoid()
                                            )
                                        def forward(self, x):
                                            return self.model(x)

                                    # Initialize models and optimizers
                                    generator = NumGenerator(latent_dim, input_dim).to(device)
                                    discriminator = NumDiscriminator(input_dim).to(device)
                                    criterion = nn.BCELoss()
                                    optim_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
                                    optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

                                    # Prepare training data
                                    X = torch.tensor(scaled_data, dtype=torch.float32).to(device)
                                    batch_size = min(128, len(X))
                                    data_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)

                                    # Create progress bar
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    # Training loop
                                    for epoch in range(num_epochs):
                                        g_running, d_running = 0.0, 0.0
                                        batches = 0
                                        
                                        for real_data in data_loader:
                                            batches += 1
                                            current_batch = real_data.size(0)
                                            valid = torch.ones(current_batch, 1).to(device)
                                            fake = torch.zeros(current_batch, 1).to(device)

                                            # Train Discriminator
                                            optim_D.zero_grad()
                                            pred_real = discriminator(real_data)
                                            loss_real = criterion(pred_real, valid)
                                            noise = torch.randn(current_batch, latent_dim).to(device)
                                            fake_data = generator(noise)
                                            pred_fake = discriminator(fake_data.detach())
                                            loss_fake = criterion(pred_fake, fake)
                                            loss_D = (loss_real + loss_fake) / 2
                                            loss_D.backward()
                                            optim_D.step()

                                            # Train Generator
                                            optim_G.zero_grad()
                                            noise = torch.randn(current_batch, latent_dim).to(device)
                                            fake_data = generator(noise)
                                            pred_fake = discriminator(fake_data)
                                            loss_G = criterion(pred_fake, valid)
                                            loss_G.backward()
                                            optim_G.step()
                                            
                                            g_running += loss_G.item()
                                            d_running += loss_D.item()
                                        
                                        # Update progress
                                        progress_bar.progress((epoch + 1) / num_epochs)
                                        status_text.text(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {d_running/batches:.4f}, Loss G: {g_running/batches:.4f}")
                                    
                                    st.success("Vanilla GAN training complete!")
                                    
                                    # Generate synthetic samples
                                    generator.eval()
                                    with torch.no_grad():
                                        noise = torch.randn(num_samples, latent_dim).to(device)
                                        synthetic_tensor = generator(noise)
                                        synthetic = synthetic_tensor.cpu().numpy()
                                        
                                    # Inverse transform to original scale
                                    synthetic = scaler.inverse_transform(synthetic)

                                # Create DataFrame with synthetic data
                                synthetic_df = pd.DataFrame(synthetic, columns=numeric_cols)

                                # Display synthetic data
                                st.subheader("Generated Synthetic Data")
                                st.dataframe(synthetic_df.head(10))

                                # Add download button for synthetic data
                                csv_synthetic = synthetic_df.to_csv(index=False)
                                st.download_button(
                                    "Download Synthetic Data", 
                                    csv_synthetic,
                                    file_name="synthetic_data.csv",
                                    mime="text/csv"
                                )

                                # Combined PCA visualization (2D) for both original and synthetic data
                                st.subheader("PCA Visualization: Original vs Synthetic Data")
                                combined_data = np.vstack([original, synthetic])
                                pca = PCA(n_components=2)
                                pca_result = pca.fit_transform(combined_data)
                                n_original = original.shape[0]
                                original_pca = pca_result[:n_original]
                                synthetic_pca = pca_result[n_original:]

                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.scatter(original_pca[:, 0], original_pca[:, 1], color='blue', label="Original Data", s=20, alpha=0.7)
                                ax.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], color='red', label="Synthetic Data", s=20, alpha=0.7)
                                ax.set_xlabel("Principal Component 1")
                                ax.set_ylabel("Principal Component 2")
                                ax.set_title("PCA: Original vs Synthetic Data")
                                ax.legend()
                                st.pyplot(fig)

                                # t-SNE visualization
                                st.subheader("t-SNE Visualization: Original vs Synthetic Data")
                                tsne = TSNE(n_components=2, random_state=42)
                                tsne_result = tsne.fit_transform(combined_data)
                                original_tsne = tsne_result[:n_original]
                                synthetic_tsne = tsne_result[n_original:]

                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.scatter(original_tsne[:, 0], original_tsne[:, 1], 
                                          color='blue', label="Original Data", s=20, alpha=0.7)
                                ax.scatter(synthetic_tsne[:, 0], synthetic_tsne[:, 1], 
                                          color='red', label="Synthetic Data", s=20, alpha=0.7)
                                ax.set_xlabel("t-SNE Component 1")
                                ax.set_ylabel("t-SNE Component 2")
                                ax.set_title("t-SNE: Original vs Synthetic Data")
                                ax.legend()
                                st.pyplot(fig)

                                # Statistical comparison
                                st.subheader("Statistical Comparison")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("Original Data Statistics")
                                    st.dataframe(pd.DataFrame(original, columns=numeric_cols).describe())
                                with col2:
                                    st.write("Synthetic Data Statistics")
                                    st.dataframe(synthetic_df.describe())
                                
                                # Feature-by-feature distribution comparison using KDE
                                st.subheader("Feature Distribution Comparison (PDF)")
                                for i, col in enumerate(numeric_cols):
                                    if i % 2 == 0:
                                        cols = st.columns(2)
                                    
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    # Calculate KDE for original data
                                    kde_orig = stats.gaussian_kde(original[:, i])
                                    x_range = np.linspace(min(original[:, i].min(), synthetic[:, i].min()),
                                                         max(original[:, i].max(), synthetic[:, i].max()), 100)
                                    ax.plot(x_range, kde_orig(x_range), 'b-', label='Original', linewidth=2)
                                    
                                    # Calculate KDE for synthetic data
                                    kde_syn = stats.gaussian_kde(synthetic[:, i])
                                    ax.plot(x_range, kde_syn(x_range), 'r-', label='Synthetic', linewidth=2)
                                    
                                    ax.set_title(f'Probability Density: {col}')
                                    ax.set_xlabel('Value')
                                    ax.set_ylabel('Density')
                                    ax.legend()
                                    cols[i % 2].pyplot(fig)

                                # Calculate Jensen-Shannon Divergence for each feature
                                st.subheader("Distribution Similarity (Jensen-Shannon Divergence)")
                                js_divergences = []

                                for i, col in enumerate(numeric_cols):
                                    # Get probability distributions using KDE
                                    kde_orig = stats.gaussian_kde(original[:, i])
                                    kde_syn = stats.gaussian_kde(synthetic[:, i])
                                    
                                    # Evaluate both distributions on a common support
                                    x_range = np.linspace(min(original[:, i].min(), synthetic[:, i].min()),
                                                         max(original[:, i].max(), synthetic[:, i].max()), 100)
                                    p = kde_orig(x_range)
                                    q = kde_syn(x_range)
                                    
                                    # Normalize to ensure they sum to 1
                                    p = p / p.sum()
                                    q = q / q.sum()
                                    
                                    # Calculate JS divergence
                                    m = 0.5 * (p + q)
                                    js_div = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))
                                    js_divergences.append((col, js_div))

                                # Display JS divergences in a nice format
                                js_df = pd.DataFrame(js_divergences, columns=['Feature', 'JS Divergence'])
                                js_df = js_df.sort_values('JS Divergence')

                                # Plot JS divergences
                                fig, ax = plt.subplots(figsize=(10, 4))
                                sns.barplot(data=js_df, x='Feature', y='JS Divergence', ax=ax)
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)

                                # Show numeric values
                                st.write("Jensen-Shannon Divergence for each feature (lower is better):")
                                st.dataframe(js_df.style.background_gradient(cmap='RdYlGn_r'))

                                # Add quality metrics calculation
                                st.subheader("Synthetic Data Quality Metrics")

                                # Calculate Inception Score
                                def calculate_inception_score(data, n_splits=10):
                                    scores = []
                                    n_samples = data.shape[0]
                                    split_size = n_samples // n_splits
                                    
                                    for i in range(n_splits):
                                        split = data[i * split_size:(i + 1) * split_size]
                                        # Calculate softmax probabilities
                                        probs = np.exp(split) / np.sum(np.exp(split), axis=1, keepdims=True)
                                        # Calculate p(y)
                                        p_yx = np.mean(probs, axis=0)
                                        # Calculate KL divergence
                                        kl = np.sum(probs * (np.log(probs + 1e-10) - np.log(p_yx + 1e-10)), axis=1)
                                        # Calculate exponential of expected KL
                                        scores.append(np.exp(np.mean(kl)))
                                    
                                    return np.mean(scores), np.std(scores)

                                # Calculate FID score
                                def calculate_fid(real_data, synthetic_data):
                                    # Calculate statistics for real and synthetic data
                                    mu1, sigma1 = real_data.mean(axis=0), np.cov(real_data, rowvar=False)
                                    mu2, sigma2 = synthetic_data.mean(axis=0), np.cov(synthetic_data, rowvar=False)
                                    
                                    # Calculate squared difference between means
                                    diff = mu1 - mu2
                                    
                                    # Calculate matrix sqrt
                                    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
                                    
                                    # Check and correct imaginary numbers from sqrt
                                    if np.iscomplexobj(covmean):
                                        covmean = covmean.real
                                    
                                    # Calculate FID
                                    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
                                    return fid

                                # Calculate and display metrics
                                is_score, is_std = calculate_inception_score(synthetic)
                                fid_score = calculate_fid(original, synthetic)

                                col1, col2 = st.columns(2)
                                 
                                with col2:
                                    st.metric(
                                        "Fréchet Inception Distance (lower is better)", 
                                        f"{fid_score:.2f}",
                                        help="Measures similarity between real and generated data distributions"
                                    )

                                # Plot correlation matrices comparison
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                sns.heatmap(pd.DataFrame(original, columns=numeric_cols).corr(), 
                                            ax=ax1, cmap='coolwarm', vmin=-1, vmax=1)
                                ax1.set_title('Original Data Correlations')
                                sns.heatmap(synthetic_df.corr(), 
                                            ax=ax2, cmap='coolwarm', vmin=-1, vmax=1)
                                ax2.set_title('Synthetic Data Correlations')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                            except Exception as e:
                                st.error(f"Error: {e}")
                else:
                    st.warning("No numerical columns found.")
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

        else:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(image)
                
                if denoising_method == "Median Filter":
                    denoised_input = Image.fromarray(img_array).filter(ImageFilter.MedianFilter(size=filter_size))
                    denoised_input = np.array(denoised_input)
                
                
                st.image(denoised_input, caption="Input Image", use_container_width=True)
                
                st.session_state['gan_input_data'] = Image.fromarray(denoised_input)
                
                with st.expander("Image statistics", expanded=True):
                    dims = img_array.shape
                    st.write(f"Dimensions: {dims}")
                    if img_array.ndim == 3:
                        cols = st.columns(img_array.shape[2])
                        for i, ch in enumerate(('R', 'G', 'B')[:dims[2]]):
                            vals = img_array[:, :, i].ravel()
                            cols[i].metric(f"{ch} mean", f"{vals.mean():.1f}")
                
                with st.expander("GAN: generate cartoon image", expanded=True):
                    cartoon_options = st.columns(2)
                    with cartoon_options[0]:
                        knn_neighbors = st.slider("KNN Neighbors", 2, 100, 4)
                    with cartoon_options[1]:
                        missing_rate = st.slider("Missing Rate", 0.1, 0.9, 0.8, 0.05)
                    
                    if st.button("Generate Cartoon Image"):
                        st.info("Generating cartoon-style image...")
                        try:
                            cartoon_img = generate_cartoon_image(st.session_state['gan_input_data'])
                            st.success("Cartoon image generated!")
                            
                            if denoising_method == "Median Filter":
                                denoised_img = Image.fromarray(cartoon_img).filter(ImageFilter.MedianFilter(size=filter_size))
                                denoised_img = np.array(denoised_img)
                            
                            col1, col2 = st.columns(2)
                            col1.image(cartoon_img, caption="Raw Cartoon Output", use_container_width=True)
                            col2.image(denoised_img, caption=f"Denoised with {denoising_method}", use_container_width=True)
                            
                            st.subheader("KNN Imputation Results")
                            st.write(f"Using {knn_neighbors} neighbors with {missing_rate:.2f} missing rate")
                            
                            imputed_img = knn_image_imputation(denoised_img, 
                                                             missing_rate=missing_rate, 
                                                             n_neighbors=knn_neighbors)
                            
                            st.image(imputed_img, caption="Final KNN-Imputed Result", use_container_width=True)
                            
                            pil_final = Image.fromarray(imputed_img)
                            buf = io.BytesIO()
                            pil_final.save(buf, format="PNG")
                            byte_im = buf.getvalue()
                            
                            st.download_button(
                                label="Download Final Image",
                                data=byte_im,
                                file_name="cartoon_result.png",
                                mime="image/png"
                            )
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"Error processing image: {e}")

with pages[1]:
    st.header("About")
    st.write("""    
    **Upload CSVs or images to generate data using GAN models**
    
    - **CSV**: extracts numerical columns and generates synthetic data using CTGAN or vanilla GAN
    - **Image**: extracts pixel data and generates cartoon-style images with denoising and KNN imputation
    
    This application demonstrates the power of generative models for both tabular and image data.
    """)
    st.markdown("---")
    st.subheader("Developer")
    st.image("developer.jpg", caption="Soumya Savarn", width=80)
    st.write("Soumya Savarn, Data Science & AI, IIT Guwahati")
    st.write("[GitHub](https://github.com/soumyasavarn) | [LinkedIn](https://www.linkedin.com/in/soumya-savarn-3483b2165/?originalSubdomain=in)")
