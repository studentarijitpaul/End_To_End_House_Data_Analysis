const navToggle = document.querySelector("[data-nav-toggle]");
const navigation = document.querySelector("[data-nav]");

if (navToggle && navigation) {
    navToggle.addEventListener("click", () => {
        const isOpen = navigation.classList.toggle("is-open");
        navToggle.setAttribute("aria-expanded", String(isOpen));
    });
}

const predictionForm = document.querySelector("[data-predict-form]");

if (predictionForm) {
    predictionForm.addEventListener("submit", (event) => {
        if (!predictionForm.checkValidity()) {
            event.preventDefault();
            predictionForm.reportValidity();
        }
    });
}
