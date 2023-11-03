
let signup = document.querySelector(".signup-choose");
let login = document.querySelector(".login-choose");
let formSection = document.querySelector(".form-section");
let loginForm = document.getElementById("login-form")
let signupForm = document.getElementById("signup-form")
let loginButton = document.getElementById("b1")
let sinupButton = document.getElementById("b2")

const py_server_address = localStorage.getItem("py_server_address")

signup.addEventListener("click", () => {
    formSection.classList.add("form-section-move");
});

login.addEventListener("click", () => {
    formSection.classList.remove("form-section-move");
});

loginButton.addEventListener("click", loginUser)
sinupButton.addEventListener("click", registerUser)

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

async function requestModel(user_id) { 
    console.log("req model called")
    const text = await sendJson({name: "My Model"}, `http://${py_server_address}/add_model/${user_id}`)
    console.log("ModEL id: ", text)
    return text
}

async function loginUser() {
    console.log(loginForm)
    let errorArea = document.querySelector(".login-box").querySelector(".error-area");;
    let UsernameInputElement = loginForm.querySelector('input[name="login"]');
    let PasswordInputElement = loginForm.querySelector('input[name="password"]');

    errorArea.innerHTML = "";
    UsernameInputElement.classList.remove('error-alert')
    PasswordInputElement.classList.remove('error-alert')

    if (UsernameInputElement.value.length == 0) {
        errorArea.innerHTML = "Fill the username field"
        UsernameInputElement.classList.add('error-alert')
        return
    }

    const userLoginData = buildJsonFormData(loginForm)
    try {
        let login_data_response = await sendJson(userLoginData, `http://${py_server_address}/login_user`);
        const responseJson = JSON.parse(login_data_response);
        if (responseJson.error) {
            console.error("Ошибка при авторизации пользователя:", responseJson.error);
            errorArea.innerHTML = responseJson.error;
            switch (responseJson["problemPart"]){
                case "username":
                    UsernameInputElement.classList.add("error-alert");
                    break;
                case "password":
                    PasswordInputElement.classList.add("error-alert");
                    break;
            }
        } else {
            console.log("Пользователь авторизирован, ID: ", responseJson.user_id);
            localStorage.setItem("user_id", responseJson.user_id)
            localStorage.setItem("model_id", await requestModel(responseJson.user_id))
            window.location = "../templates/main.html";
        }
    } catch (error) {
        console.error("Произошла ошибка при отправке запроса:", error);
        errorArea.innerHTML = `Internal Error. You can try again, but it's unlikely to work...`
    }
}

async function registerUser() {
    let errorArea = signupForm.querySelector(".error-area");
    let UsernameInputElement = signupForm.querySelector('input[name="login"]');
    let EmailInputElement = signupForm.querySelector('input[name="mail"]');
    let PasswordInputElement = signupForm.querySelector('input[name="password"]');
    let PasswordConfirmInputElement = signupForm.querySelector('input[name="password-confirm"]');

    // Удаляем ошибки, выведенные на прошлой попытке регистрации
    UsernameInputElement.classList.remove('error-alert')
    EmailInputElement.classList.remove('error-alert')
    errorArea.innerHTML = "";

    if (UsernameInputElement.value.length == 0) {
        errorArea.innerHTML = "Fill the username field"
        UsernameInputElement.classList.add('error-alert')
        return
    }
    if (EmailInputElement.value.length == 0) {
        errorArea.innerHTML = "Fill the email field"
        EmailInputElement.classList.add('error-alert')
        return
    }
    // Проверяем пароль
    if (PasswordInputElement.value.length < 6) {
        errorArea.innerHTML = "Password length must be at least 6 symbols";
        return;
    }
    if (PasswordInputElement.value !== PasswordConfirmInputElement.value) {
        errorArea.innerHTML = "Password confirmation doesn't match password";
        return;
    }


    const userRegistrationData = buildJsonFormData(signupForm);
    try {
        let add_data_response = await sendJson(userRegistrationData, `http://${py_server_address}/add_user`);
        const responseJson = JSON.parse(add_data_response);
        if (responseJson.error) {
            console.error("Ошибка при регистрации пользователя:", responseJson.error);
            errorArea.innerHTML = responseJson.error;
            switch (responseJson["problemPart"]){
                case "username":
                    UsernameInputElement.classList.add("error-alert");
                    break;
                case "mail":
                    EmailInputElement.classList.add("error-alert");
                    break;
            }
        } else {
            console.log("Пользователь зарегистрирован, ID: ", responseJson.user_id);
            user_id = localStorage.setItem("user_id", responseJson.user_id)
            localStorage.setItem("model_id", await requestModel(responseJson.user_id))
            window.location = "../templates/main.html";
        }
    } catch (error) {
        console.error("Произошла ошибка при отправке запроса:", error);
        errorArea.innerHTML = `Internal Error. You can try again, but it's unlikely to work...`
    }
}