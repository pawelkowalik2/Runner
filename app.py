import streamlit as st
import pandas as pd
import datetime
import os
import instructor
#############################################################################
import re

def time_input_with_placeholder(label, placeholder="MM:SS", default=None):
    # Zamień domyślną wartość na tekst, jeśli podana
    default_str = ""
    if isinstance(default, datetime.time):
        default_str = default.strftime("%M:%S")

    # Input tekstowy z placeholderem
    value = st.text_input(
        label,
        value=default_str,
        placeholder=placeholder
    )

    # Walidacja: dopasowanie do formatu MM:SS lub HH:MM:SS
    pattern = r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$"
    match = re.match(pattern, value)

    if match:
        parts = [int(p) if p else 0 for p in match.groups()]
        h, m, s = 0, parts[0], parts[1]  # domyślnie MM:SS
        if len(parts) == 3 and match.groups()[2] is not None:
            h, m, s = parts  # HH:MM:SS
        try:
            t = datetime.time(hour=h, minute=m, second=s)
            return t
        except ValueError:
            st.error("Niepoprawny zakres czasu")
            return None
    elif value:
        st.error("Podaj czas w formacie MM:SS lub HH:MM:SS")
        return None
    return None
#############################################################################
from pycaret.regression import load_model, predict_model
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, Literal
from langfuse import observe
from langfuse.openai import OpenAI as LangfuseOpenAI

load_dotenv(os.path.join('model', '.env'))

openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
MODEL_NAME = os.path.join('model', 'best_halfmarathon_model')

@st.cache_data
def get_model():
        return load_model(MODEL_NAME)

st.title('Runner 🏃')
st.markdown('### Wypełniając tych kilka poniższych informacji możesz z dość dużym prawdopodobieństwem dowiedzieć się jaki będzie twój czas w półmaratonie.')

st.markdown('## Podaj swoje dane')

tab1, tab2 = st.tabs(['Ręcznie', 'Porozmawiaj ze mną'])

with tab1:

    gender = ['Mężczyzna','Kobieta']

    genderSelection = st.radio(
        "Jaka jest Twoja płeć?",
        gender,
    )

    if genderSelection != None:
        if genderSelection == 'Mężczyzna':
            genderSelection = 'M'
        else:
            genderSelection = 'K'

    ageSelection = st.number_input(
        'W którym roku się urodziłeś?',
        1900,
        2025,
        1994
    )

    user_5km_time_input = time_input_with_placeholder(
        "Podaj swój czas na 5 km",
        placeholder="MM:SS"
    )

    # st.write('Twoj czas na 5km to:', user_5km_time_input)

    if(user_5km_time_input != None):
        user_5km_time_input_in_sec = user_5km_time_input.hour*3600 + user_5km_time_input.minute*60 + user_5km_time_input.second

        tempo = (user_5km_time_input_in_sec / 5) / 60

        # st.write('Twój czas w sekundach to: ', user_5km_time_input_in_sec)
        st.write('Twoje tempo ( min/km ) to: ', tempo)

        prediction_data_df = pd.DataFrame([{
            'Płeć': genderSelection,
            'Rocznik': ageSelection,
            '5 km Tempo': tempo
        }])

        model = get_model()

        prediction_df = predict_model(model, prediction_data_df)
        predicted_time = pd.to_datetime(prediction_df['prediction_label'][0], unit='s').strftime('%H:%M:%S')

        st.write('Twój przewidywany czas w najbliższym półmaratonie to: ', predicted_time)

with tab2:

    llm_client = LangfuseOpenAI()

    class UserInfo(BaseModel):
        gender: Optional[Literal['M', 'K']] = None
        year_of_birth: Optional[int] = None
        how_many_minutes: Optional[float] = None

    instructor_openai_client = instructor.from_openai(llm_client)

    @observe(name='user info extraction')
    def extractUserInfo(user_input: str) -> UserInfo:
        return instructor_openai_client.chat.completions.create(
            # model="gpt-4o-mini",
            model="gpt-4o",
            response_model=UserInfo,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': '''Zbierz dane od użytkownika. Obecnie mamy rok 2025.
                            Płeć (`gender`) zwróć jako dokładnie jedną literę:
- 'M' jeśli użytkownik jest mężczyzną.
- 'K' jeśli użytkownik jest kobietą.
- None jeśli nie można określić.

Jeżeli użytkownik nie poda płci wprost, spróbuj ustalić płeć na podstawie imienia 
(np. imiona kończące się na "a" są zwykle żeńskie, z wyjątkami jak 'Kuba' czy 'Barnaba').

Przykłady:
- 'Jestem Paweł' → gender = 'M'
- 'Mam na imię Anna' → gender = 'K'
- 'Nazywam się Kuba' → gender = 'M''
'''
                        },
                        {
                            'type': 'text',
                            'text': user_input
                        }
                    ]
                }
            ]
        )

    if 'user_data' not in st.session_state:
        st.session_state.user_data = UserInfo()

    st.write('Po prostu powiedz mi jak masz na imię, kiedy się urodziłeś i jaki jest twój czas na 5km - i patrz jak dzieje się magia :smile:')
    
    user_input = st.chat_input('przedstaw się')

    if user_input is not None:
        st.write(user_input)

        new_data = extractUserInfo(user_input)

        for field, value in new_data.model_dump().items():
            if getattr(st.session_state.user_data, field) is None and value is not None:
                setattr(st.session_state.user_data, field, value)

        field_labels = {
            "gender": "informacji o twojej płci",
            "year_of_birth": "twojego roku urodzenia",
            "how_many_minutes": "twojego czasu na 5 km"
        }

        missing_fields = [
            field_labels[field]
            for field, value in st.session_state.user_data.model_dump().items()
            if value is None
        ]

        if missing_fields:
            st.write(f"Brakuje jeszcze: {', '.join(missing_fields)}")
        else:
            st.write("Wszystkie dane zebrane ✅")
            # st.write(st.session_state.user_data.gender, st.session_state.user_data.year_of_birth, st.session_state.user_data.how_many_minutes/5)
            
            prediction_data_df = pd.DataFrame([{
                'Płeć': st.session_state.user_data.gender,
                'Rocznik': st.session_state.user_data.year_of_birth,
                '5 km Tempo': st.session_state.user_data.how_many_minutes/5
            }])

            model = get_model()

            prediction_df = predict_model(model, prediction_data_df)
            predicted_time = pd.to_datetime(prediction_df['prediction_label'][0], unit='s').strftime('%H:%M:%S')

            st.write('Twój przewidywany czas w najbliższym półmaratonie to: ', predicted_time)