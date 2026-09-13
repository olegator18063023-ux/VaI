import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from engine import (
    HORIZON,
    LABELS,
    SENSORS,
    UNITS,
    WINDOW,
    generate_unit,
    local_explanation,
    make_features,
    outside_training_range,
    simulate,
    train_models,
)


st.set_page_config(
    page_title="РЕСУРС / AI",
    page_icon="⚙️",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                ellipse at 90% 0%,
                #dbece8 0%,
                transparent 48%
            ),
            linear-gradient(135deg, #f8f5ed, #eef2ee);
        color: #163b37;
    }

    h1, h2, h3 {
        font-family: Georgia, "Times New Roman", serif !important;
        letter-spacing: -0.035em;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #cddbd5;
        border-top: 4px solid #007f73;
        padding: 18px;
        border-radius: 6px;
    }

    section[data-testid="stSidebar"] {
        background: #e4ece6;
    }

    .eyebrow {
        font-family: monospace;
        color: #007f73;
        letter-spacing: 0.18em;
        font-size: 12px;
    }

    @keyframes enter {
        from {
            opacity: 0;
            transform: translateY(8px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    h1 {
        animation: enter 0.55s ease-out;
    }

    @media (prefers-reduced-motion: reduce) {
        h1 {
            animation: none;
        }
    }

    @media (max-width: 700px) {
        h1 {
            font-size: 2rem !important;
        }

        [data-testid="stMetric"] {
            padding: 10px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Генерация телеметрии и обучение моделей...")
def load_bundle():
    return train_models()


@st.cache_data
def demo_fleet():
    return {
        "Станок №1": generate_unit(1001, 1001),
        "Насос №2": generate_unit(1002, 1002),
        "Компрессор №3": generate_unit(1003, 1003),
    }


def reset_controls():
    st.session_state["temperature_delta"] = 0.0
    st.session_state["vibration_delta"] = 0.0
    st.session_state["pressure_delta"] = 0.0


bundle = load_bundle()
fleet = demo_fleet()


with st.sidebar:
    st.markdown("### Пульт инженера")

    equipment = st.selectbox(
        "Оборудование",
        list(fleet.keys()),
    )

    full_history = fleet[equipment]
    last_available = int(full_history["hour"].iloc[-2])

    current_hour = st.slider(
        "Наработка, ч",
        min_value=WINDOW - 1,
        max_value=last_available,
        value=int(last_available * 0.6),
        key=f"hour_{equipment}",
    )

    st.divider()

    enabled = st.toggle(
        "Сломать систему",
        value=False,
    )

    st.caption(
        "Изменение плавно нарастает за последние 12 часов. "
        "Ползунки задают отклонение в текущий момент."
    )

    temperature_delta = st.slider(
        "Изменение температуры, °C",
        min_value=-15.0,
        max_value=45.0,
        value=0.0,
        step=1.0,
        key="temperature_delta",
        disabled=not enabled,
    )

    vibration_delta = st.slider(
        "Изменение вибрации, мм/с",
        min_value=-2.0,
        max_value=8.0,
        value=0.0,
        step=0.1,
        key="vibration_delta",
        disabled=not enabled,
    )

    pressure_delta = st.slider(
        "Изменение давления масла, бар",
        min_value=-3.0,
        max_value=1.5,
        value=0.0,
        step=0.1,
        key="pressure_delta",
        disabled=not enabled,
    )

    st.button(
        "Сбросить воздействия",
        on_click=reset_controls,
    )

    st.divider()

    st.caption(
        "Синтетический стенд. Данные предназначены "
        "для демонстрации работы модели."
    )


observed = full_history.loc[
    full_history["hour"] <= current_hour,
    ["hour", *SENSORS],
].copy()


scenario = simulate(
    observed,
    temperature_delta if enabled else 0.0,
    vibration_delta if enabled else 0.0,
    pressure_delta if enabled else 0.0,
)


base_features = make_features(observed).tail(1)
scenario_features = make_features(scenario).tail(1)


base_rul_array, base_risk_array = bundle.predict(base_features)
rul_array, risk_array = bundle.predict(scenario_features)


base_rul = float(base_rul_array[0])
base_risk = float(base_risk_array[0])
rul = float(rul_array[0])
risk = float(risk_array[0])


resource_percent = 100 * rul / max(current_hour + rul, 1)


if risk >= 0.65 or rul <= 24:
    status = "Критический износ"
    message = (
        "Рекомендация: приоритетная диагностика оборудования."
    )
    severity = "error"

elif risk >= 0.25 or rul <= 72:
    status = "Предупреждение"
    message = (
        "Рекомендация: запланировать осмотр "
        "и проверку датчиков."
    )
    severity = "warning"

else:
    status = "Зона нормы"
    message = (
        "По модели выраженных признаков "
        "близкого отказа нет."
    )
    severity = "success"


st.markdown(
    '<div class="eyebrow">'
    "PREDICTIVE MAINTENANCE / ENGINEERING DEMO"
    "</div>",
    unsafe_allow_html=True,
)

st.title("РЕСУРС / AI")

st.write(
    f"**{equipment}** · "
    f"Наработка: **{current_hour} ч** · "
    f"Горизонт риска: **{HORIZON} ч**"
)


if enabled:
    st.info(
        "Активен сценарий изменения телеметрии. "
        "Это проверка реакции модели, а не точный физический "
        "расчет будущего отказа."
    )


if severity == "error":
    st.error(f"{status}. {message}")

elif severity == "warning":
    st.warning(f"{status}. {message}")

else:
    st.success(f"{status}. {message}")


column_1, column_2, column_3 = st.columns(3)


column_1.metric(
    "Остаточный ресурс",
    f"{rul:.1f} ч",
    f"{rul - base_rul:+.1f} ч" if enabled else None,
)


column_2.metric(
    "Оставшаяся доля жизни",
    f"{resource_percent:.1f}%",
)


column_3.metric(
    f"Риск отказа за {HORIZON} ч",
    f"{risk:.1%}",
    f"{100 * (risk - base_risk):+.1f} п.п."
    if enabled
    else None,
    delta_color="inverse",
)


st.caption(
    "Доля жизни рассчитывается как RUL / (наработка + RUL). "
    "Это демонстрационный показатель, а не процент "
    "паспортного ресурса оборудования."
)


outliers = outside_training_range(
    bundle,
    scenario_features.iloc[0],
)


if outliers:
    st.warning(
        "Некоторые признаки вышли за диапазон обучения. "
        "Прогноз в таком сценарии может быть ненадежным."
    )

    with st.expander("Признаки вне диапазона обучения"):
        st.write(", ".join(outliers))

else:
    st.caption(
        "Признаки находятся внутри диапазонов, "
        "которые использовались при обучении."
    )


telemetry_tab, explanation_tab, quality_tab = st.tabs(
    [
        "Телеметрия",
        "Почему такой прогноз",
        "Качество и методика",
    ]
)


with telemetry_tab:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            LABELS[sensor]
            for sensor in SENSORS
        ],
    )

    for row_number, sensor in enumerate(SENSORS, start=1):
        figure.add_trace(
            go.Scatter(
                x=observed["hour"],
                y=observed[sensor],
                name="Исходная телеметрия",
                legendgroup="base",
                showlegend=row_number == 1,
                line={
                    "color": "#007f73",
                    "width": 2,
                },
            ),
            row=row_number,
            col=1,
        )

        if enabled:
            tail = scenario.tail(WINDOW)

            figure.add_trace(
                go.Scatter(
                    x=tail["hour"],
                    y=tail[sensor],
                    name="Сценарий",
                    legendgroup="scenario",
                    showlegend=row_number == 1,
                    line={
                        "color": "#de713d",
                        "width": 3,
                    },
                ),
                row=row_number,
                col=1,
            )

        figure.update_yaxes(
            title_text=UNITS[sensor],
            row=row_number,
            col=1,
        )

    figure.update_xaxes(
        title_text="Наработка, ч",
        row=3,
        col=1,
    )

    figure.update_layout(
        height=680,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={
            "l": 15,
            "r": 15,
            "t": 45,
            "b": 20,
        },
        legend={
            "orientation": "h",
            "y": 1.09,
        },
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    current = pd.DataFrame(
        {
            "Датчик": [
                LABELS[sensor]
                for sensor in SENSORS
            ],
            "Значение": [
                round(
                    float(scenario[sensor].iloc[-1]),
                    2,
                )
                for sensor in SENSORS
            ],
            "Единица": [
                UNITS[sensor]
                for sensor in SENSORS
            ],
        }
    )

    st.dataframe(
        current,
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Скачать видимую телеметрию CSV",
        data=scenario.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="telemetry_scenario.csv",
        mime="text/csv",
    )


with explanation_tab:
    st.subheader(
        "Вклад датчиков в текущую вероятность отказа"
    )

    contributions, reference_risk, explained_risk = (
        local_explanation(
            bundle,
            scenario_features.iloc[0],
        )
    )

    local = pd.DataFrame(
        {
            "Датчик": [
                LABELS[sensor]
                for sensor in SENSORS
            ],
            "Вклад, п.п.": contributions * 100,
        }
    ).sort_values("Вклад, п.п.")

    real_values = local["Вклад, п.п."].astype(float)

    minimum_marker = 0.15

    display_values = real_values.apply(
        lambda value: (
            minimum_marker
            if 0 <= value < minimum_marker
            else -minimum_marker
            if -minimum_marker < value < 0
            else value
        )
    )

    colors = [
        "#de713d" if value > 0 else "#007f73"
        for value in real_values
    ]

    explanation_figure = go.Figure()

    explanation_figure.add_trace(
        go.Bar(
            x=display_values,
            y=local["Датчик"],
            orientation="h",
            marker_color=colors,
            text=[
                f"{value:+.4f} п.п."
                for value in real_values
            ],
            textposition="outside",
            hovertemplate=[
                f"{sensor}<br>"
                f"Реальный вклад: {value:+.6f} п.п."
                "<extra></extra>"
                for sensor, value in zip(
                    local["Датчик"],
                    real_values,
                )
            ],
        )
    )

    explanation_figure.add_vline(
        x=0,
        line_color="#163b37",
    )

    max_abs = max(
        float(np.max(np.abs(real_values))),
        0.5,
    )

    explanation_figure.update_xaxes(
        range=[
            -max_abs * 1.25,
            max_abs * 1.25,
        ],
        title_text=(
            "Вклад в вероятность отказа, п.п."
        ),
        zeroline=True,
        zerolinecolor="#163b37",
    )

    explanation_figure.update_layout(
        height=360,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={
            "l": 10,
            "r": 100,
            "t": 25,
            "b": 45,
        },
        showlegend=False,
    )

    st.plotly_chart(
        explanation_figure,
        use_container_width=True,
    )

    st.dataframe(
        local.style.format(
            {
                "Вклад, п.п.": "{:+.6f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    if np.max(np.abs(contributions)) < 0.0001:
        st.info(
            "На текущей наработке влияние отдельных "
            "датчиков относительно здорового эталона "
            "очень маленькое. Точные значения показаны "
            "в таблице."
        )

    st.write(
        f"Риск эталона: **{reference_risk:.1%}**. "
        f"Сумма вкладов: "
        f"**{100 * contributions.sum():+.2f} п.п.**. "
        f"Текущий риск: **{explained_risk:.1%}**."
    )

    st.caption(
        "График показывает локальное объяснение текущего "
        "прогноза. Вклад считается относительно здорового "
        "эталона из обучающей выборки."
    )

    with st.expander(
        "Глобальная важность признаков модели"
    ):
        importance = pd.DataFrame(
            {
                "Признак": bundle.columns,
                "Важность": (
                    bundle.classifier.feature_importances_
                ),
            }
        ).sort_values("Важность")

        importance_figure = px.bar(
            importance,
            x="Важность",
            y="Признак",
            orientation="h",
            color_discrete_sequence=["#007f73"],
        )

        importance_figure.update_layout(
            height=460,
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            importance_figure,
            use_container_width=True,
        )

        st.caption(
            "Глобальная важность показывает, какие признаки "
            "важны для модели в целом. Это не объяснение "
            "одного конкретного прогноза."
        )


with quality_tab:
    st.subheader(
        "Проверка на невиденных экземплярах"
    )

    metrics = pd.DataFrame(
        {
            "Метрика": list(bundle.metrics.keys()),
            "Значение": [
                (
                    f"{value:.4f}"
                    if isinstance(value, float)
                    else str(value)
                )
                for value in bundle.metrics.values()
            ],
        }
    )

    st.dataframe(
        metrics,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        """
        **Как устроен эксперимент**

        - 80 независимых синтетических жизненных циклов.
        - 52 узла используются для обучения.
        - 14 узлов используются для калибровки.
        - 14 узлов используются для тестирования.
        - Один шаг телеметрии равен одному часу.
        - Окно признаков содержит последние 12 измерений.
        - RUL равен времени до синтетического отказа.
        - Классификация отвечает на вопрос:
          будет ли отказ в течение следующих 24 часов.
        - Истинный RUL и будущие измерения не используются
          как признаки модели.
        """
    )

    if not enabled:
        true_rul_values = full_history.loc[
            full_history["hour"] == current_hour,
            "rul",
        ]

        if not true_rul_values.empty:
            true_rul = float(true_rul_values.iloc[0])

            st.info(
                f"Истинный RUL синтетического стенда: "
                f"{true_rul:.0f} ч. "
                f"Абсолютная ошибка прогноза: "
                f"{abs(rul - true_rul):.1f} ч."
            )

    else:
        st.info(
            "Для сценария ручного повреждения истинный RUL "
            "не пересчитывается, потому что ползунки изменяют "
            "телеметрию, но не физический жизненный цикл."
        )


st.divider()

st.caption(
    "РЕСУРС / AI · Конкурсный прототип · "
    "Не использовать для управления промышленным оборудованием."
)
