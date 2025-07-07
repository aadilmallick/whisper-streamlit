import streamlit as st

from pydantic import BaseModel, Field, computed_field
from typing import List, LiteralString, Literal, Callable, TypeVar, Any

type FieldType = Literal["text", "number", "date", "time", "radio", "checkbox", "select", "textarea"]

T = TypeVar("T")

class STState(BaseModel):
    default_state: dict[str, Any] = Field(default={})

    def init_state(self):
        for (key, value) in self.default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def get_state(key: str) -> Any:
        return st.session_state[key]
    
    @staticmethod
    def set_state(key: str, value: Any):
        st.session_state[key] = value


class STForm(BaseModel):
    clear_on_submit: bool = Field(default=True)
    fields: dict[FieldType, str] = Field(default={})
    key: str

    def create_form(self, on_submit: Callable[[dict[str, any]], None]):
        fieldsDict : dict[str, any] = {}
        with st.form(self.key, clear_on_submit=self.clear_on_submit):
            for field in self.fields.keys():
                placeholder = self.fields[field]
                if field == "text":
                    fieldsDict[placeholder] = st.text_input(placeholder)
                elif field == "number":
                    fieldsDict[placeholder] = st.number_input(placeholder)
                elif field == "date":
                    fieldsDict[placeholder] = st.date_input(placeholder)
                elif field == "time":
                    fieldsDict[placeholder] = st.time_input(placeholder)
                elif field == "radio":
                    fieldsDict[placeholder] = st.radio(placeholder)
                elif field == "checkbox":
                    fieldsDict[placeholder] = st.checkbox(placeholder)
                elif field == "select":
                    fieldsDict[placeholder] = st.selectbox(placeholder)
                elif field == "textarea":
                    fieldsDict[placeholder] = st.text_area(placeholder)
            if st.form_submit_button("Submit"):
                on_submit(fieldsDict)
        return fieldsDict

def on_click_fn(name: str):
    st.write(f"hello {name}")

def main():
    st.title("Hello from pythonai!")
    stateManager = STState(default_state={
        "is_submitted": False
    })
    stateManager.init_state()
    form = STForm(fields={
        "text": "Name",
        "number": "Age"
    }, key="unique form key")
    fields_dict = form.create_form(on_submit=lambda x: stateManager.set_state("is_submitted", True))
    st.write(fields_dict)
    st.button("click me", on_click=on_click_fn, args=("Aadil",)) 

    if stateManager.get_state("is_submitted"):
        st.success("form submitted")
        st.balloons()

    [tab1, tab2] = st.tabs(["Tab 1", "Tab 2"])
    with tab1:
        st.write("This is tab 1")
    with tab2:
        st.write("This is tab 2")
    
    [col1, col2] = st.columns(2)
    with col1:
        st.write("This is column 1")
    with col2:
        st.write("This is column 2")
    
    container = st.container(border=True)
    with container:
        st.write("This is a container")
    
    placeholder = st.empty()
    placeholder.write("This is a placeholder 1")
    placeholder.write("This is a placeholder 2")
    toggles()

@st.fragment()
def toggles():
    with st.expander("This is an expander"):
        st.write("This is an expander")
    
    with st.expander("This is an expander", expanded=True):
        st.write("This is an expander")
        st.button("refresh component", on_click=lambda: st.rerun(scope="fragment"))
    
    with st.expander("This is an expander", expanded=False):
        st.write("This is an expander")


if __name__ == "__main__":
    main()
