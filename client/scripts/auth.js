
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

async function loginUser() {
    console.log(loginForm)
    let errorArea = document.querySelector(".login-box").querySelector(".error-area");;
    let UsernameInputElement = loginForm.querySelector('input[name="login"]');
    let PasswordInputElement = loginForm.querySelector('input[name="password"]');

    errorArea.innerHTML = "";
    UsernameInputElement.classList.remove('error-alert')
    PasswordInputElement.classList.remove('error-alert')

    const userLoginData = buildJsonFormData(loginForm)
    try {
        let login_data_response = await sendJson(userLoginData, `http://${py_server_port}/login_user`);
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
        }
    } catch (error) {
        console.error("Произошла ошибка при отправке запроса:", error);
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
        let add_data_response = await sendJson(userRegistrationData, `http://${py_server_port}/add_user`);
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
        }
    } catch (error) {
        console.error("Произошла ошибка при отправке запроса:", error);
    }
}