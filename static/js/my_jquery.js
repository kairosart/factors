

$(document).ready(function(e){
    $("#submitButton").on('click',function(){

        $(this).val("Wait a second...");
        $(this).css('color','red');
    });
	$("#submitButton1").on('click',function(){

        $(this).val("Calculating...");
        $(this).css('color','red');
    });    
});