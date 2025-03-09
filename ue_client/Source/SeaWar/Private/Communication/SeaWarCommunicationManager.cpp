// 创建TCP客户端连接到Python服务器
bool USeaWarCommunicationManager::ConnectToServer(const FString& Host, int32 Port)
{
    Socket = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateSocket(NAME_Stream, TEXT("SeaWarSocket"), false);
    
    FIPv4Address IPAddress;
    FIPv4Address::Parse(Host, IPAddress);
    TSharedRef<FInternetAddr> Addr = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
    Addr->SetIp(IPAddress.Value);
    Addr->SetPort(Port);
    
    bool Connected = Socket->Connect(*Addr);
    if (Connected)
    {
        // 启动接收线程
        ReceiveThread = FRunnableThread::Create(new FSeaWarSocketReceiver(Socket, this), TEXT("SeaWarReceiver"));
        UE_LOG(LogTemp, Display, TEXT("Connected to Python server at %s:%d"), *Host, Port);
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to connect to Python server at %s:%d"), *Host, Port);
    }
    
    return Connected;
}

// 发送命令到Python服务器
bool USeaWarCommunicationManager::SendCommand(const FString& CommandJson)
{
    if (!Socket || !Socket->IsConnected())
    {
        UE_LOG(LogTemp, Error, TEXT("Socket not connected"));
        return false;
    }
    
    FString MessageWithNewline = CommandJson + TEXT("\n");
    TCHAR* Serialized = MessageWithNewline.GetCharArray().GetData();
    int32 Size = FCString::Strlen(Serialized);
    int32 BytesSent = 0;
    
    bool Success = Socket->Send((uint8*)TCHAR_TO_UTF8(Serialized), Size, BytesSent);
    
    return Success && BytesSent == Size;
}

// 处理从Python服务器接收到的游戏状态
void USeaWarCommunicationManager::HandleGameState(const TSharedPtr<FJsonObject>& JsonObject)
{
    // 解析实体数据
    TArray<TSharedPtr<FJsonValue>> EntitiesJson = JsonObject->GetArrayField(TEXT("entities"));
    
    // 清除旧实体
    GameState->ClearEntities();
    
    // 添加新实体
    for (auto& EntityJson : EntitiesJson)
    {
        TSharedPtr<FJsonObject> EntityObj = EntityJson->AsObject();
        
        FString EntityId = EntityObj->GetStringField(TEXT("id"));
        FString EntityType = EntityObj->GetStringField(TEXT("type"));
        FString Faction = EntityObj->GetStringField(TEXT("faction"));
        
        // 获取位置
        TSharedPtr<FJsonObject> PosObj = EntityObj->GetObjectField(TEXT("position"));
        FVector Position(
            PosObj->GetNumberField(TEXT("x")),
            PosObj->GetNumberField(TEXT("y")),
            PosObj->GetNumberField(TEXT("z"))
        );
        
        // 创建或更新实体
        ASeaWarEntity* Entity = GameState->GetEntityById(EntityId);
        if (!Entity)
        {
            // 创建新实体
            Entity = SpawnEntityByType(EntityType, Position);
            if (Entity)
            {
                Entity->SetEntityId(EntityId);
                Entity->SetFaction(Faction);
                GameState->AddEntity(Entity);
            }
        }
        else
        {
            // 更新现有实体
            Entity->SetActorLocation(Position);
        }
        
        // 更新其他属性...
    }
    
    // 通知UI更新
    OnGameStateUpdated.Broadcast();
}