requirejs['jquery'];

window.currentSource = 0;
window.attentions = null;

function rgb2hex(component) {
    let component_hex = component.toString(16);
    if (component_hex.length === 1) {
	component_hex = "0" + component_hex;
    }
    return component_hex;
}

// note : use the global window.attentions variable, which is a 2d
// array of shape (source, target)
function refreshAttentions() {
    
    if (window.attentions === null) {
	return;
    }

    $('.target').each(function (i, element) {
	// [0->1]
	let attention = window.attentions[window.currentSource][i];
	// [205->0]
	let red = Math.floor(205 - attention * 205);
	let red_hex = rgb2hex(red);
	// [255->0]
	let green = Math.floor(255 - attention * 255);
	let green_hex = rgb2hex(green);
	$(element).css("background-color", "#" + red_hex + green_hex + "ff");
    });
    
}
window.refreshAttentions = refreshAttentions;


$('.source').on('mouseenter', function(e) {
    $(this).css('background-color', 'cyan');
    window.currentSource = $('.source').index(this);
    window.refreshAttentions();
});

$('.source').on('mouseleave', function(e) {
    $(this).css('background-color', 'transparent');
});

