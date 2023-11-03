
// Функция для изменения SVG на другое SVG
function changeSVGIcon(container, newSVG, SVGStyleClass="") {
    const currentIcon = container.querySelector('.svg-icon');
    currentIcon.remove();
    container.innerHTML += newSVG;
    container.querySelector('.svg-icon').setAttribute("class", SVGStyleClass)
}