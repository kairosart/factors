

$(document).ready(function(e){
    $("#submitButton").on('click',function(){

        $(this).val("Wait a second...");
        $(this).css('color','white');
        $(this).css('background-color', 'red');
    });
	$("#submitButton1").on('click',function(){
		$(this).html('Calculating...');
        $(this).css('color','white');
        $(this).css('background-color', 'red');
    });
    $("#submitButton2").on('click',function(){
		$(this).html('Downloading...');
        $(this).css('color','white');
        $(this).css('background-color', 'red');
    });
    $('input[type=radio]').on('click', function(){
        href = $(this).val();
        $('#form').attr("action", href)
    });
});