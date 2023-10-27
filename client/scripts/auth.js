
let signup = document.querySelector(".signup-choose");
let login = document.querySelector(".login-choose");
let formSection = document.querySelector(".form-section");
let loginForm = document.getElementById("login-form")
let signupForm = document.getElementById("signup-form")

const py_server_port = "localhost:3000"

signup.addEventListener("click", () => {
    formSection.classList.add("form-section-move");
});

login.addEventListener("click", () => {
    formSection.classList.remove("form-section-move");
});



function buildJsonFormData(form) {
    const jsonFormData = {}
    for (const pair of new FormData(form)) {
        jsonFormData[pair[0]] = pair[1]
    }
    return jsonFormData
}

async function sendJson(sending_object, url) {

    const resp = await fetch(url, {
        method: 'POST',
        mode: 'cors',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify(sending_object)
    })

    return await resp.text()
}

function loginUser() {
    console.log(loginForm)
    const userLoginData = buildJsonFormData(loginForm)
    console.log(userLoginData)
}

function registerUser(){
    console.log(signupForm)
    const userRegistrationData = buildJsonFormData(signupForm)
    console.log(userRegistrationData)
    let add_data_response = sendJson(userRegistrationData, `http://${py_server_port}/add_user`)
    console.log(add_data_response)
}
