
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)


# Nastavitve modela in omejitve videoposnetka
MODEL_ID = "Vansh180/VideoMae-ffc23-deepfake-detector"

NAJVECJA_VELIKOST_DATOTEKE = 100 * 1024 * 1024
NAJDALJSE_TRAJANJE = 180
STEVILO_SLICIC = 16


# Nastavitve spletne aplikacije
st.set_page_config(
    page_title="Detektor videoposnetkov z umetno inteligenco",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource
def nalozi_model():
    """Naloži vnaprej naučeni model za analizo videoposnetkov."""

    procesor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)

    naprava = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    try:
        model = VideoMAEForVideoClassification.from_pretrained(
            MODEL_ID
        )
        model = model.to(naprava)

    except Exception:
        # Če modela ni mogoče naložiti na MPS,
        # poskusi z uporabo procesorja CPU.
        naprava = "cpu"

        model = VideoMAEForVideoClassification.from_pretrained(
            MODEL_ID
        )
        model = model.to(naprava)

    model.eval()

    return model, procesor, naprava


def pridobi_slicice(pot_do_videa, stevilo=STEVILO_SLICIC):
    """Enakomerno izlušči sličice iz videoposnetka."""

    zajem = cv2.VideoCapture(pot_do_videa)

    if not zajem.isOpened():
        raise ValueError(
            "Videoposnetka ni mogoče odpreti."
        )

    try:
        fps = zajem.get(cv2.CAP_PROP_FPS)
        skupno_slicic = int(
            zajem.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        sirina = int(
            zajem.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        visina = int(
            zajem.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if fps <= 0 or skupno_slicic <= 0:
            raise ValueError(
                "Videoposnetek ne vsebuje veljavnih sličic."
            )

        trajanje = skupno_slicic / fps

        if trajanje > NAJDALJSE_TRAJANJE:
            raise ValueError(
                "Videoposnetek je daljši od 180 sekund."
            )

        if max(sirina, visina) > 3840:
            raise ValueError(
                "Ločljivost videoposnetka presega 4K."
            )

        indeksi = np.linspace(
            0,
            skupno_slicic - 1,
            min(stevilo, skupno_slicic),
            dtype=int,
        )

        slicice = []

        for indeks in indeksi:
            zajem.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(indeks),
            )

            uspesno, slica = zajem.read()

            if uspesno:
                slica_rgb = cv2.cvtColor(
                    slica,
                    cv2.COLOR_BGR2RGB,
                )

                slicice.append(
                    Image.fromarray(slica_rgb)
                )

        if not slicice:
            raise ValueError(
                "Iz videoposnetka ni bilo mogoče pridobiti sličic."
            )

        return slicice, trajanje, sirina, visina

    finally:
        zajem.release()


def analiziraj_videoposnetek(
    model,
    procesor,
    slicice,
    naprava,
):
    """Analizira zaporedje sličic kot video vhod za model."""

    zahtevano_stevilo = getattr(
        model.config,
        "num_frames",
        STEVILO_SLICIC,
    )

    # Izbere zahtevano število sličic.
    indeksi = np.linspace(
        0,
        len(slicice) - 1,
        min(zahtevano_stevilo, len(slicice)),
        dtype=int,
    )

    izbrane_slicice = [
        slicice[i].convert("RGB")
        for i in indeksi
    ]

    # Model VideoMAE običajno pričakuje določeno število sličic.
    # Če jih je premalo, ponovi zadnjo sličico.
    while len(izbrane_slicice) < zahtevano_stevilo:
        izbrane_slicice.append(izbrane_slicice[-1])

    obdelani_podatki = procesor(
        images=izbrane_slicice,
        return_tensors="pt",
    )

    slikovni_podatki = obdelani_podatki["pixel_values"]

    # Pričakovana oblika je:
    # (serija, sličice, kanali, višina, širina)
    if slikovni_podatki.ndim == 4:
        slikovni_podatki = slikovni_podatki.unsqueeze(0)

    if slikovni_podatki.ndim != 5:
        raise ValueError(
            "Model je prejel nepričakovano obliko podatkov: "
            f"{tuple(slikovni_podatki.shape)}."
        )

    if slikovni_podatki.shape[1] != zahtevano_stevilo:
        raise ValueError(
            "Število sličic se ne ujema s pričakovanji modela. "
            f"Pričakovano: {zahtevano_stevilo}, "
            f"prejeto: {slikovni_podatki.shape[1]}."
        )

    slikovni_podatki = slikovni_podatki.to(naprava)

    with torch.inference_mode():
        izhod = model(
            pixel_values=slikovni_podatki
        )

        verjetnosti = torch.softmax(
            izhod.logits,
            dim=-1,
        )[0]

    oznake = model.config.id2label
    rezultati = []

    for indeks, verjetnost in enumerate(
        verjetnosti.tolist()
    ):
        oznaka = oznake.get(
            indeks,
            oznake.get(str(indeks), str(indeks)),
        )

        rezultati.append({
            "Oznaka modela": str(oznaka),
            "Ocena modela": float(verjetnost),
        })

    return rezultati


def poisci_oznako_ponaredka(rezultati):
    """Poišče razred, ki jasno označuje ponarejen video."""

    izrazi = (
        "fake",
        "deepfake",
        "manipulated",
        "synthetic",
        "generated",
        "forged",
    )

    for rezultat in rezultati:
        oznaka = (
            rezultat["Oznaka modela"]
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        if any(izraz in oznaka for izraz in izrazi):
            return rezultat

    return None


# Uporabniški vmesnik
st.title(
    "🎬 Detektor videoposnetkov, ustvarjenih z umetno inteligenco"
)

st.caption(
    "Analiza videoposnetkov z uporabo vnaprej naučenega "
    "modela strojnega učenja."
)

st.warning(
    "Model je namenjen predvsem prepoznavanju ponarejenih "
    "oziroma manipuliranih videoposnetkov. Ni potrjeno, da "
    "zanesljivo prepozna vse videoposnetke, ustvarjene z "
    "umetno inteligenco. Rezultati so informativni in ne "
    "predstavljajo dokončnega dokaza o izvoru videoposnetka."
)

nalozena_datoteka = st.file_uploader(
    "Izberi videoposnetek",
    type=["mp4", "mov", "avi", "mkv"],
)

if nalozena_datoteka is not None:

    if nalozena_datoteka.size > NAJVECJA_VELIKOST_DATOTEKE:
        st.error(
            "Datoteka presega dovoljeno velikost 100 MB."
        )
        st.stop()

    st.video(nalozena_datoteka.getvalue())

    if st.button(
        "Analiziraj videoposnetek",
        type="primary",
    ):
        pot_do_datoteke = None

        try:
            pripona = Path(
                nalozena_datoteka.name
            ).suffix.lower()

            with tempfile.NamedTemporaryFile(
                suffix=pripona,
                delete=False,
            ) as zacasna_datoteka:

                zacasna_datoteka.write(
                    nalozena_datoteka.getvalue()
                )

                pot_do_datoteke = zacasna_datoteka.name

            with st.spinner(
                "Pridobivam sličice iz videoposnetka ..."
            ):
                (
                    slicice,
                    trajanje,
                    sirina,
                    visina,
                ) = pridobi_slicice(
                    pot_do_datoteke
                )

            with st.spinner(
                "Nalagam model za analizo videoposnetka ..."
            ):
                model, procesor, naprava = nalozi_model()

            with st.spinner(
                "Analiziram videoposnetek ..."
            ):
                rezultati = analiziraj_videoposnetek(
                    model,
                    procesor,
                    slicice,
                    naprava,
                )

            st.subheader("Rezultati analize")

            stolpec1, stolpec2, stolpec3 = st.columns(3)

            stolpec1.metric(
                "Analizirane sličice",
                len(slicice),
            )

            stolpec2.metric(
                "Trajanje videoposnetka",
                f"{trajanje:.1f} s",
            )

            stolpec3.metric(
                "Ločljivost",
                f"{sirina} × {visina}",
            )

            st.write(
                f"**Uporabljena naprava:** {naprava}"
            )

            oznaka_ponaredka = poisci_oznako_ponaredka(
                rezultati
            )

            if oznaka_ponaredka is not None:
                ocena = oznaka_ponaredka["Ocena modela"]

                st.metric(
                    "Ocena razreda "
                    + oznaka_ponaredka["Oznaka modela"],
                    f"{ocena * 100:.1f} %",
                )

                st.caption(
                    "To je ocena modela za izbrani razred, "
                    "ne pa potrjena verjetnost, da je celoten "
                    "videoposnetek ustvarila umetna inteligenca."
                )

            else:
                st.info(
                    "Med oznakami modela ni bilo mogoče "
                    "zanesljivo prepoznati razreda za ponarejen "
                    "videoposnetek. Prikazani so vsi razredi, "
                    "ne da bi ugibali njihov pomen."
                )

            tabela_rezultatov = pd.DataFrame(rezultati)

            st.subheader("Rezultati po razredih")

            st.dataframe(
                tabela_rezultatov,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Graf ocen modela")

            st.bar_chart(
                tabela_rezultatov.set_index(
                    "Oznaka modela"
                )["Ocena modela"]
            )

            st.subheader("Omejitve analize")

            st.markdown(
                """
                - Model ni univerzalni detektor vseh
                  videoposnetkov, ustvarjenih z umetno inteligenco.
                - Stiskanje, montaža in uporabljena metoda
                  generiranja lahko vplivajo na rezultat.
                - Model je treba preveriti na različnih resničnih
                  in umetno ustvarjenih videoposnetkih.
                - Rezultat modela sam po sebi ne dokazuje,
                  ali je videoposnetek pristen ali ponarejen.
                """
            )

        except Exception as napaka:
            st.error(
                f"Analiza ni uspela: "
                f"{type(napaka).__name__}: {napaka}"
            )

        finally:
            if (
                pot_do_datoteke is not None
                and os.path.exists(pot_do_datoteke)
            ):
                os.remove(pot_do_datoteke)
