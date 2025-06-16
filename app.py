import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
from utils.data_processing import process_uploaded_data, validate_data
from utils.plotting import create_plot, get_plot_types, get_color_palettes
from utils.statistics import calculate_statistics, generate_comparison_stats

# Konfiguracja strony
st.set_page_config(
    page_title="Narzędzie Wizualizacji Treningu GRU",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicjalizacja session state
if 'training_data' not in st.session_state:
    st.session_state.training_data = {}
if 'plot_config' not in st.session_state:
    st.session_state.plot_config = {
        'plot_type': 'line',
        'color_palette': 'viridis',
        'show_markers': True,
        'line_width': 2
    }

def main():
    st.title("�� Narzędzie Wizualizacji Treningu GRU")
    st.markdown("Interaktywne narzędzie do analizy i wizualizacji metryk treningu sieci neuronowych GRU")
    
    # Sidebar dla kontrolek
    with st.sidebar:
        st.header("⚙️ Ustawienia")
        
        # Sekcja wczytywania danych
        st.subheader("�� Dane Treningowe")
        data_source = st.radio(
            "Źródło danych:",
            ["Wczytaj plik", "Wprowadź ręcznie"],
            help="Wybierz sposób wprowadzenia danych treningowych"
        )
        
        if data_source == "Wczytaj plik":
            uploaded_file = st.file_uploader(
                "Wybierz plik CSV lub JSON",
                type=['csv', 'json'],
                help="Plik powinien zawierać kolumny z metrykami treningu (epoch, loss, accuracy, val_loss, val_accuracy, itp.)"
            )
            
            if uploaded_file is not None:
                try:
                    df = process_uploaded_data(uploaded_file)
                    if validate_data(df):
                        st.session_state.training_data['main'] = df
                        st.success(f"✅ Wczytano {len(df)} rekordów")
                        st.dataframe(df.head())
                    else:
                        st.error("❌ Nieprawidłowy format danych")
                except Exception as e:
                    st.error(f"❌ Błąd podczas wczytywania: {str(e)}")
        
        else:  # Wprowadź ręcznie
            st.subheader("✍️ Ręczne wprowadzanie danych")
            
            # Formularz do wprowadzania danych
            with st.form("manual_data_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    epochs = st.number_input("Liczba epok", min_value=1, max_value=1000, value=10)
                    initial_loss = st.number_input("Początkowa strata", min_value=0.0, value=2.0, step=0.1)
                    final_loss = st.number_input("Końcowa strata", min_value=0.0, value=0.1, step=0.01)
                
                with col2:
                    initial_acc = st.number_input("Początkowa dokładność", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
                    final_acc = st.number_input("Końcowa dokładność", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
                    noise_level = st.slider("Poziom szumu", 0.0, 0.5, 0.1)
                
                if st.form_submit_button("�� Generuj dane treningowe"):
                    # Generowanie syntetycznych danych treningowych
                    epoch_data = np.arange(1, epochs + 1)
                    
                    # Generowanie krzywej straty z szumem
                    loss_trend = np.exp(-epoch_data * 0.1) * (initial_loss - final_loss) + final_loss
                    loss_noise = np.random.normal(0, noise_level, len(epoch_data))
                    loss_data = np.maximum(loss_trend + loss_noise, 0.001)
                    
                    # Generowanie krzywej dokładności z szumem
                    acc_trend = 1 - np.exp(-epoch_data * 0.1) * (1 - initial_acc)
                    acc_trend = acc_trend * (final_acc / acc_trend[-1])  # Skalowanie do końcowej wartości
                    acc_noise = np.random.normal(0, noise_level * 0.1, len(epoch_data))
                    acc_data = np.clip(acc_trend + acc_noise, 0, 1)
                    
                    # Generowanie danych walidacyjnych (nieco gorsze niż treningowe)
                    val_loss_data = loss_data * (1 + np.random.uniform(0.05, 0.2, len(epoch_data)))
                    val_acc_data = acc_data * (1 - np.random.uniform(0.01, 0.05, len(epoch_data)))
                    
                    manual_df = pd.DataFrame({
                        'epoch': epoch_data,
                        'loss': loss_data,
                        'accuracy': acc_data,
                        'val_loss': val_loss_data,
                        'val_accuracy': val_acc_data
                    })
                    
                    st.session_state.training_data['main'] = manual_df
                    st.success(f"✅ Wygenerowano dane dla {epochs} epok")
        
        # Dodawanie dodatkowych sesji treningowych dla porównania
        st.subheader("�� Porównanie sesji")
        if st.button("➕ Dodaj sesję porównawczą"):
            if 'main' in st.session_state.training_data:
                session_name = f"sesja_{len(st.session_state.training_data)}"
                # Kopiuj główne dane z drobnymi modyfikacjami
                base_df = st.session_state.training_data['main'].copy()
                # Dodaj niewielkie losowe modyfikacje
                for col in base_df.columns:
                    if col != 'epoch' and col in base_df.select_dtypes(include=[np.number]).columns:
                        noise = np.random.normal(0, base_df[col].std() * 0.1, len(base_df))
                        base_df[col] = base_df[col] + noise
                        if 'accuracy' in col:
                            base_df[col] = np.clip(base_df[col], 0, 1)
                        elif 'loss' in col:
                            base_df[col] = np.maximum(base_df[col], 0.001)
                
                st.session_state.training_data[session_name] = base_df
                st.success(f"✅ Dodano {session_name}")
        
        # Lista aktywnych sesji
        if st.session_state.training_data:
            st.write("**Aktywne sesje:**")
            for session_name in st.session_state.training_data.keys():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {session_name}")
                with col2:
                    if session_name != 'main' and st.button("��️", key=f"del_{session_name}"):
                        del st.session_state.training_data[session_name]
                        st.rerun()
    
    # Główna obszar aplikacji
    if st.session_state.training_data:
        # Tabs dla różnych funkcjonalności
        tab1, tab2, tab3, tab4 = st.tabs(["�� Wizualizacja", "�� Statystyki", "⚙️ Ustawienia wykresów", "�� Eksport"])
        
        with tab1:
            st.header("Wizualizacja metryk treningowych")
            
            # Kontrolki wykresów
            col1, col2, col3 = st.columns(3)
            
            with col1:
                available_columns = []
                for df in st.session_state.training_data.values():
                    available_columns.extend([col for col in df.columns if col != 'epoch'])
                available_columns = list(set(available_columns))
                
                selected_metrics = st.multiselect(
                    "Wybierz metryki do wyświetlenia:",
                    available_columns,
                    default=available_columns[:2] if len(available_columns) >= 2 else available_columns,
                    help="Wybierz które metryki mają być wyświetlone na wykresie"
                )
            
            with col2:
                plot_type = st.selectbox(
                    "Typ wykresu:",
                    get_plot_types(),
                    help="Wybierz typ wykresu do wyświetlenia danych"
                )
            
            with col3:
                show_comparison = st.checkbox(
                    "Porównaj sesje",
                    value=len(st.session_state.training_data) > 1,
                    help="Pokaż wszystkie sesje na jednym wykresie"
                )
            
            if selected_metrics:
                # Tworzenie wykresów
                if len(selected_metrics) == 1:
                    fig = create_plot(
                        st.session_state.training_data,
                        selected_metrics[0],
                        plot_type,
                        st.session_state.plot_config,
                        show_comparison
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Tworzenie subplot dla wielu metryk
                    rows = (len(selected_metrics) + 1) // 2
                    fig = make_subplots(
                        rows=rows, cols=2,
                        subplot_titles=selected_metrics,
                        vertical_spacing=0.08
                    )
                    
                    for i, metric in enumerate(selected_metrics):
                        row = (i // 2) + 1
                        col = (i % 2) + 1
                        
                        for session_name, df in st.session_state.training_data.items():
                            if metric in df.columns:
                                fig.add_trace(
                                    go.Scatter(
                                        x=df['epoch'],
                                        y=df[metric],
                                        mode='lines+markers' if st.session_state.plot_config['show_markers'] else 'lines',
                                        name=f"{metric} ({session_name})" if show_comparison else metric,
                                        line=dict(width=st.session_state.plot_config['line_width']),
                                        showlegend=(i == 0)
                                    ),
                                    row=row, col=col
                                )
                    
                    fig.update_layout(height=300 * rows, title_text="Porównanie metryk treningowych")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Wyświetlanie danych w tabeli
            if st.checkbox("�� Pokaż dane szczegółowe"):
                selected_session = st.selectbox(
                    "Wybierz sesję:",
                    list(st.session_state.training_data.keys())
                )
                st.dataframe(st.session_state.training_data[selected_session])
        
        with tab2:
            st.header("Analiza statystyczna")
            
            # Statystyki dla poszczególnych sesji
            for session_name, df in st.session_state.training_data.items():
                with st.expander(f"�� Statystyki dla {session_name}", expanded=(session_name == 'main')):
                    stats = calculate_statistics(df)
                    
                    # Wyświetlanie statystyk w kolumnach
                    cols = st.columns(len(stats))
                    for i, (metric, stat_dict) in enumerate(stats.items()):
                        with cols[i % len(cols)]:
                            st.metric(
                                label=f"{metric} - Średnia",
                                value=f"{stat_dict['mean']:.4f}",
                                delta=f"±{stat_dict['std']:.4f}"
                            )
                            st.metric(
                                label=f"{metric} - Min/Max",
                                value=f"{stat_dict['min']:.4f}",
                                delta=f"{stat_dict['max']:.4f}"
                            )
            
            # Porównanie sesji
            if len(st.session_state.training_data) > 1:
                st.subheader("�� Porównanie sesji")
                comparison_stats = generate_comparison_stats(st.session_state.training_data)
                st.dataframe(comparison_stats)
        
        with tab3:
            st.header("Ustawienia wykresów")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.plot_config['color_palette'] = st.selectbox(
                    "Paleta kolorów:",
                    get_color_palettes(),
                    index=get_color_palettes().index(st.session_state.plot_config['color_palette'])
                )
                
                st.session_state.plot_config['show_markers'] = st.checkbox(
                    "Pokaż markery",
                    value=st.session_state.plot_config['show_markers']
                )
            
            with col2:
                st.session_state.plot_config['line_width'] = st.slider(
                    "Grubość linii:",
                    1, 5,
                    st.session_state.plot_config['line_width']
                )
                
                # Opcje stylizacji
                custom_title = st.text_input("Niestandardowy tytuł wykresu:", "")
                if custom_title:
                    st.session_state.plot_config['custom_title'] = custom_title
        
        with tab4:
            st.header("Eksport danych i wykresów")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("�� Eksport wykresów")
                if selected_metrics:
                    export_format = st.selectbox("Format eksportu:", ["PNG", "SVG", "PDF", "HTML"])
                    export_width = st.number_input("Szerokość (px):", min_value=400, max_value=2000, value=800)
                    export_height = st.number_input("Wysokość (px):", min_value=300, max_value=1500, value=600)
                    
                    if st.button("�� Eksportuj wykres"):
                        # Tworzenie wykresu do eksportu
                        fig = create_plot(
                            st.session_state.training_data,
                            selected_metrics[0],
                            plot_type,
                            st.session_state.plot_config,
                            show_comparison
                        )
                        
                        if export_format == "PNG":
                            img_bytes = fig.to_image(format="png", width=export_width, height=export_height)
                            st.download_button(
                                label="⬇️ Pobierz PNG",
                                data=img_bytes,
                                file_name="training_metrics.png",
                                mime="image/png"
                            )
                        elif export_format == "SVG":
                            img_bytes = fig.to_image(format="svg", width=export_width, height=export_height)
                            st.download_button(
                                label="⬇️ Pobierz SVG",
                                data=img_bytes,
                                file_name="training_metrics.svg",
                                mime="image/svg+xml"
                            )
                        elif export_format == "HTML":
                            html_str = fig.to_html()
                            st.download_button(
                                label="⬇️ Pobierz HTML",
                                data=html_str,
                                file_name="training_metrics.html",
                                mime="text/html"
                            )
            
            with col2:
                st.subheader("�� Eksport danych")
                selected_session = st.selectbox("Wybierz sesję do eksportu:", list(st.session_state.training_data.keys()))
                data_format = st.selectbox("Format danych:", ["CSV", "JSON", "Excel"])
                
                if st.button("�� Eksportuj dane"):
                    df = st.session_state.training_data[selected_session]
                    
                    if data_format == "CSV":
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Pobierz CSV",
                            data=csv,
                            file_name=f"{selected_session}_data.csv",
                            mime="text/csv"
                        )
                    elif data_format == "JSON":
                        json_str = df.to_json(orient='records', indent=2)
                        st.download_button(
                            label="⬇️ Pobierz JSON",
                            data=json_str,
                            file_name=f"{selected_session}_data.json",
                            mime="application/json"
                        )
                    elif data_format == "Excel":
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, sheet_name=selected_session, index=False)
                        excel_data = output.getvalue()
                        st.download_button(
                            label="⬇️ Pobierz Excel",
                            data=excel_data,
                            file_name=f"{selected_session}_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
    
    else:
        # Stan początkowy - brak danych
        st.info("�� Rozpocznij od wczytania danych treningowych lub wprowadzenia ich ręcznie w panelu bocznym.")
        
        # Przykładowe informacje o formatach danych
        with st.expander("ℹ️ Informacje o formatach danych"):
            st.markdown("""
            **Obsługiwane formaty danych:**
            
            **CSV:** Plik powinien zawierać nagłówki kolumn i dane liczbowe:
            ```
            epoch,loss,accuracy,val_loss,val_accuracy
            1,2.1,0.12,2.3,0.10
            2,1.8,0.25,2.0,0.22
            ...
            ```
            
            **JSON:** Dane w formacie JSON z listą obiektów:
            ```json
            [
                {"epoch": 1, "loss": 2.1, "accuracy": 0.12, "val_loss": 2.3, "val_accuracy": 0.10},
                {"epoch": 2, "loss": 1.8, "accuracy": 0.25, "val_loss": 2.0, "val_accuracy": 0.22}
            ]
            ```
            
            **Wymagane kolumny:**
            - `epoch` - numer epoki (liczba całkowita)
            - Minimum jedna metryka treningowa (np. `loss`, `accuracy`, `val_loss`, `val_accuracy`)
            """)

if __name__ == "__main__":
    main()
