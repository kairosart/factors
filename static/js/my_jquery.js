

$(document).ready(function(e){

    $("#submitButton2").on('click',function(){
		$(this).html('Downloading...');
        $(this).css('color','white');
        $(this).css('background-color', 'red');
    });


    $("#buttonTryAgain").on('click',function(){
		parent.history.back();
        return false;
    });

    $("#model_Selection").change(function () {
        $("#forecast_Time")
            .find("option")
            .show()
            .not("option[value*='" + this.value + "']").hide();

        $("#forecast_Time").val(
            $("#forecast_Time").find("option:visible:first").val());

    }).change();



    $('[data-toggle="popover"]').popover();

    $('input[name=submit]')
    .click(
         function ()
         {
             $(this).prop("disabled",true);
         }
    );
});

$(function () {
	$('#datetimepicker4').datetimepicker({
	format: 'L'
  });
});

function modal(){
   $('.modal').modal('show');
   setTimeout(function () {
    console.log('hejsan');
    $('.modal').modal('hide');
   }, 2000);

}





