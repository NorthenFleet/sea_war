void UCommandWidget::SendMoveCommand(ASeaWarEntity* Entity, const FVector& TargetLocation)
{
    if (!Entity || !CommunicationManager)
        return;
        
    // 创建移动命令JSON
    TSharedPtr<FJsonObject> CommandObj = MakeShareable(new FJsonObject);
    CommandObj->SetStringField(TEXT("type"), TEXT("command"));
    
    TSharedPtr<FJsonObject> DataObj = MakeShareable(new FJsonObject);
    DataObj->SetStringField(TEXT("command_type"), TEXT("move"));
    DataObj->SetStringField(TEXT("actor"), Entity->GetEntityId());
    
    TSharedPtr<FJsonObject> TargetObj = MakeShareable(new FJsonObject);
    TargetObj->SetNumberField(TEXT("x"), TargetLocation.X);
    TargetObj->SetNumberField(TEXT("y"), TargetLocation.Y);
    TargetObj->SetNumberField(TEXT("z"), TargetLocation.Z);
    
    DataObj->SetObjectField(TEXT("target"), TargetObj);
    CommandObj->SetObjectField(TEXT("data"), DataObj);
    
    // 序列化为字符串
    FString OutputString;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
    FJsonSerializer::Serialize(CommandObj.ToSharedRef(), Writer);
    
    // 发送命令
    CommunicationManager->SendCommand(OutputString);
}