import streamlit as st


def calculator():
    st.title("Very simple calculator")
    st.markdown("---")
    st.write("Hello, World!")

    col1, col2, col3 = st.columns(3)

    with col1:
        num1 = st.number_input("First number", step=1)
    with col2:
        num2 = st.number_input("Second number", step=1)
    with col3:
        operation = st.selectbox(
            "Operation", [
                "Add",
                "Subtract",
                "Multiply",
                "Divide"
            ])

    if st.button("Calculate"):
        if operation == "Divide" and num2 == 0:
            st.error("You cannot divide by zero!")
        else:
            result = None
            if operation == "Add":
                result = num1 + num2
            elif operation == "Subtract":
                result = num1 - num2
            elif operation == "Multiply":
                result = num1 * num2
            elif operation == "Divide":
                result = num1 / num2

            st.success(f"**Result: {result}**")


if __name__ == "__main__":
    calculator()
